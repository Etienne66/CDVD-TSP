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
def cupy_launch():
    global kernel_Correlation_rearrange, kernel_Correlation_updateOutput
    global kernel_Correlation_updateGradFirst, kernel_Correlation_updateGradSecond
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
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self,
                first,
                second):
        global kernel_Correlation_rearrange, kernel_Correlation_updateOutput
        global gpadding, gkernel_size, gmax_displacement, gstride1, gstride2, gdevice
        #first,second -> B,C,H,W
        B = cupy.int16(first.shape[0])
        in_channels = cupy.int16(first.shape[1])
        height = cupy.int32(first.shape[2])
        width = cupy.int32(first.shape[3])
        grid_radius = cupy.int16(gmax_displacement / gstride2); #4
        grid_width = cupy.int16(grid_radius * 2 + 1); #9
        out_channels = cupy.int16(grid_width * grid_width); #81
        pbottomwidth = cupy.int16(width + 2 * gpadding);
        pbottomheight = cupy.int16(height + 2 * gpadding);

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
                                                   cupy.int16(gpadding),
                                                   first.data_ptr(),
                                                   rbot0.data_ptr()))
            kernel_Correlation_rearrange(grid   = tuple([cupy.int16((n + 16 - 1) / 16), in_channels, B]),
                                         block  = tuple([16, 1, 1]),
                                         args   = (cupy.int32(width),
                                                   cupy.int32(height),
                                                   cupy.int16(in_channels),
                                                   cupy.int16(gpadding),
                                                   second.data_ptr(),
                                                   rbot1.data_ptr()))
            kernel_Correlation_updateOutput(grid       = tuple([width, height, B]),
                                            block      = tuple([32, 1, 1]),
                                            shared_mem = in_channels * 4,
                                            args       = (cupy.int32(width),
                                                          cupy.int32(height),
                                                          cupy.int16(in_channels),
                                                          cupy.int16(gpadding),
                                                          cupy.int16(gkernel_size),
                                                          cupy.int16(gmax_displacement),
                                                          cupy.int16(gstride1),
                                                          cupy.int16(gstride2),
                                                          rbot0.data_ptr(),
                                                          rbot1.data_ptr(),
                                                          output.data_ptr()))
        elif first.is_cuda == False:
            raise NotImplementedError()

        # end

        self.save_for_backward(first, rbot0, rbot1)

        return output
    # end_def_forward

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, gradOutput):
        global kernel_Correlation_updateGradFirst, kernel_Correlation_updateGradSecond
        global gpadding, gkernel_size, gmax_displacement, gstride1, gstride2, gdevice
        first, rbot0, rbot1 = self.saved_tensors
        
        B, in_channels, height, width = first.shape
        
        gradOutput = gradOutput.contiguous(); assert(gradOutput.is_cuda)

        gradFirst = first.new_zeros([B,
                                     in_channels,
                                     height,
                                     width]) if self.needs_input_grad[0] else None
        gradSecond = first.new_zeros([B,
                                      in_channels,
                                      height,
                                      width]) if self.needs_input_grad[1] else None

        pixels = in_channels * height * width

        if first.is_cuda:
            if gradFirst is not None:
                for intSample in range(B):
                    kernel_Correlation_updateGradFirst(grid=tuple([int((pixels + 512 - 1) / 512), 1, 1]),
                                                       block=tuple([512, 1, 1]),
                                                       args=(cupy.int32(width),
                                                             cupy.int32(height),
                                                             cupy.int16(in_channels),
                                                             cupy.int16(gpadding),
                                                             cupy.int16(gkernel_size),
                                                             cupy.int16(gmax_displacement),
                                                             cupy.int16(gstride1),
                                                             cupy.int16(gstride2),
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
                                                              cupy.int16(gpadding),
                                                              cupy.int16(gkernel_size),
                                                              cupy.int16(gmax_displacement),
                                                              cupy.int16(gstride1),
                                                              cupy.int16(gstride2),
                                                              cupy.int32(pixels),
                                                              cupy.int16(intSample),
                                                              rbot0.data_ptr(),
                                                              gradOutput.data_ptr(),
                                                              gradSecond.data_ptr()))
        elif first.is_cuda == False:
            raise NotImplementedError()
        # end

        return gradFirst, gradSecond
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
                 device = 'cuda'):
        super(ModuleCorrelation, self).__init__()
        global gpadding, gkernel_size, gmax_displacement, gstride1, gstride2, gdevice
        gpadding = padding
        gkernel_size = kernel_size
        gmax_displacement = max_displacement
        gstride1 = stride1
        gstride2 = stride2
        gdevice = device
        grid_radius = max_displacement / stride2; #4
        grid_width = grid_radius * 2 + 1; #9
        out_channels = grid_width * grid_width; #81
        cupy_launch()
        
        #self.FunctionCorrelation = _FunctionCorrelation()
    # end__init__

    def forward(self, x):
        tensorFirst  = x[:,0,:,:,:]
        tensorSecond = x[:,1,:,:,:]
        return _FunctionCorrelation.apply(tensorFirst,
                                          tensorSecond)
    # end_forward
# end_ModuleCorrelation
