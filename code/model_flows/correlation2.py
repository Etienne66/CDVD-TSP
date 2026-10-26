# original from https://github.com/sniklaus/pytorch-pwc/
import torch

import cupy
import re
from pathlib import Path

class Stream:
    """ begin added by csbhr    
    """
    ptr = torch.cuda.current_stream().cuda_stream
# end

@cupy.memoize(for_each_device=True)
def cupy_launch(mixed_precision=False):
    global kernel_Correlation_rearrange, kernel_Correlation_updateOutput
    global kernel_Correlation_updateGradFirst, kernel_Correlation_updateGradSecond
    #if mixed_precision:
    if False:
        module = cupy.RawModule(code=Path('./model_flows/correlation.16.cu').read_text())
    else:
        module = cupy.RawModule(code=Path('./model_flows/correlation.32.cu').read_text())
    kernel_Correlation_rearrange = module.get_function('kernel_Correlation_rearrange')
    kernel_Correlation_updateOutput = module.get_function('kernel_Correlation_updateOutput')
    kernel_Correlation_updateGradFirst = module.get_function('kernel_Correlation_updateGradFirst')
    kernel_Correlation_updateGradSecond = module.get_function('kernel_Correlation_updateGradSecond')
# end


class _FunctionCorrelation(torch.autograd.Function):
    """ Duplicate class from `correlation.py` of
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    @staticmethod
    #@torch.amp.custom_fwd
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx,
                first,
                second,
                padding,
                kernel_size,
                max_displacement,
                stride1,
                stride2):
        global kernel_Correlation_rearrange, kernel_Correlation_updateOutput
        ctx.padding = padding
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        #first,second -> B,C,H,W
        B = cupy.int16(first.shape[0])
        in_channels = cupy.int16(first.shape[1])
        height = cupy.int32(first.shape[2])
        width = cupy.int32(first.shape[3])
        grid_radius = cupy.int16(ctx.max_displacement / stride2); #4
        grid_width = cupy.int16(grid_radius * 2 + 1); #9
        out_channels = cupy.int16(grid_width * grid_width); #81
        pbottomwidth = cupy.int16(width + 2 * padding);
        pbottomheight = cupy.int16(height + 2 * padding);

        #rbot0,rbot1  -> B,H+2*padding,W+2*padding,C
        rbot0 =  first.new_zeros([B, pbottomheight, pbottomwidth, in_channels])
        rbot1 = second.new_zeros([B, pbottomheight, pbottomwidth, in_channels])

        first = first.contiguous(); assert(first.is_cuda)
        second = second.contiguous(); assert(second.is_cuda)

        output = first.new_zeros([B, out_channels, height, width])

        if first.is_cuda:
            n = width * height
            kernel_Correlation_rearrange(grid   = tuple([cupy.int16((n + 16 - 1) / 16), in_channels, B]),
                                         block  = tuple([16, 1, 1]),
                                         args   = (cupy.int32(width),
                                                   cupy.int32(height),
                                                   cupy.int16(in_channels),
                                                   cupy.int16(padding),
                                                   first.data_ptr(),
                                                   rbot0.data_ptr()))
            kernel_Correlation_rearrange(grid   = tuple([cupy.int16((n + 16 - 1) / 16), in_channels, B]),
                                         block  = tuple([16, 1, 1]),
                                         args   = (cupy.int32(width),
                                                   cupy.int32(height),
                                                   cupy.int16(in_channels),
                                                   cupy.int16(padding),
                                                   second.data_ptr(),
                                                   rbot1.data_ptr()))
            kernel_Correlation_updateOutput(grid       = tuple([width, height, B]),
                                            block      = tuple([32, 1, 1]),
                                            shared_mem = in_channels * 4,
                                            args       = (cupy.int32(width),
                                                          cupy.int32(height),
                                                          cupy.int16(in_channels),
                                                          cupy.int16(padding),
                                                          cupy.int16(kernel_size),
                                                          cupy.int16(ctx.max_displacement),
                                                          cupy.int16(stride1),
                                                          cupy.int16(stride2),
                                                          rbot0.data_ptr(),
                                                          rbot1.data_ptr(),
                                                          output.data_ptr()))
        elif first.is_cuda == False:
            raise NotImplementedError()

        # end

        ctx.save_for_backward(first, rbot0, rbot1)

        return output
    # end_def_forward

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, gradOutput):
        global kernel_Correlation_updateGradFirst, kernel_Correlation_updateGradSecond
        first, rbot0, rbot1 = ctx.saved_tensors
        
        B, in_channels, height, width = first.shape
        
        gradOutput = gradOutput.contiguous(); assert(gradOutput.is_cuda)

        gradFirst = first.new_zeros([B,
                                     in_channels,
                                     height,
                                     width]) if ctx.needs_input_grad[0] else None
        gradSecond = first.new_zeros([B,
                                      in_channels,
                                      height,
                                      width]) if ctx.needs_input_grad[1] else None

        pixels = in_channels * height * width

        if first.is_cuda:
            if gradFirst is not None:
                for intSample in range(B):
                    kernel_Correlation_updateGradFirst(grid=tuple([int((pixels + 512 - 1) / 512), 1, 1]),
                                                       block=tuple([512, 1, 1]),
                                                       args=(cupy.int32(width),
                                                             cupy.int32(height),
                                                             cupy.int16(in_channels),
                                                             cupy.int16(ctx.padding),
                                                             cupy.int16(ctx.kernel_size),
                                                             cupy.int16(ctx.max_displacement),
                                                             cupy.int16(ctx.stride1),
                                                             cupy.int16(ctx.stride2),
                                                             cupy.int32(pixels),
                                                             cupy.int16(intSample),
                                                             rbot1.data_ptr(),
                                                             gradOutput.data_ptr(),
                                                             gradFirst.data_ptr()))
            if gradSecond is not None:
                for intSample in range(B):
                    kernel_Correlation_updateGradSecond(grid=tuple([int((pixels + 512 - 1) / 512), 1, 1]),
                                                        block=tuple([512, 1, 1]),
                                                        args=(cupy.int32(width),
                                                              cupy.int32(height),
                                                              cupy.int16(in_channels),
                                                              cupy.int16(ctx.padding),
                                                              cupy.int16(ctx.kernel_size),
                                                              cupy.int16(ctx.max_displacement),
                                                              cupy.int16(ctx.stride1),
                                                              cupy.int16(ctx.stride2),
                                                              cupy.int32(pixels),
                                                              cupy.int16(intSample),
                                                              rbot0.data_ptr(),
                                                              gradOutput.data_ptr(),
                                                              gradSecond.data_ptr()))
        elif first.is_cuda == False:
            raise NotImplementedError()
        # end

        return gradFirst, gradSecond, None, None, None, None, None
    # end_def_backward
# end_FunctionCorrelation



def FunctionCorrelation(tensorFirst, tensorSecond):
    """ Duplicate function from `correlation.py` of
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    return _FunctionCorrelation.apply(tensorFirst, tensorSecond)
# end_FunctionCorrelation

class ModuleCorrelation(torch.nn.Module):
    """ Duplicate class from `correlation.py` of
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    def __init__(self,
                 padding = 4,
                 kernel_size = 1,
                 max_displacement = 4,
                 stride1 = 1,
                 stride2 = 1,
                 device = 'cuda',
                 mixed_precision = False):
        super(ModuleCorrelation, self).__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.device = device
        grid_radius = max_displacement / stride2; #4
        grid_width = grid_radius * 2 + 1; #9
        out_channels = grid_width * grid_width; #81
        cupy_launch(mixed_precision = mixed_precision)
        
        #self.FunctionCorrelation = _FunctionCorrelation()
    # end__init__

    def compute_cost_volume(self, feat1, feat2):
        """
        only implemented for:
            kernel_size = 1
            stride1 = 1
            stride2 = 1
        """

        max_disp = self.max_displacement

        _, _, height, width = feat1.size()
        num_shifts = 2 * max_disp + 1
        feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)

        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):
                corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], dim=1, keepdim=True)
                cost_list.append(corr)
        cost_volume = torch.cat(cost_list, dim=1)
        return cost_volume

    def forward(self, x):
        tensorFirst  = x[:,0,:,:,:]
        tensorSecond = x[:,1,:,:,:]
        #print('self.max_displacement',self.max_displacement)
        if torch.cuda.is_available():
            return _FunctionCorrelation.apply(tensorFirst,
                                              tensorSecond,
                                              self.padding,
                                              self.kernel_size,
                                              self.max_displacement,
                                              self.stride1,
                                              self.stride2)
        else:
            return compute_cost_volume(tensorFirst,
                                       tensorSecond)
    # end_forward
# end_ModuleCorrelation
