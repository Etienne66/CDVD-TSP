import torch.nn as nn


###############################
# common
###############################

def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


###############################
# ResNet
###############################

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, original_model=True):
        super(ResBlock, self).__init__()
        self.original_model = original_model
        if self.original_model:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                   padding=get_same_padding(kernel_size, dilation), dilation=dilation)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                                   padding=get_same_padding(kernel_size, dilation), dilation=dilation)
            self.relu = nn.ReLU(inplace=True)

        else:
            #print("Use sequential ResBlock") 
            self.ResidualBlock = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                          padding=get_same_padding(kernel_size, dilation), dilation=dilation),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                          padding=get_same_padding(kernel_size, dilation), dilation=dilation)
            )
        
        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        
        if self.original_model:
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
        else:
            out = self.ResidualBlock(x)

        if self.res_translate is not None:
            residual = self.res_translate(residual)

        return out + residual
