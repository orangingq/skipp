from collections import OrderedDict
from typing import Optional, Type, Union
from torch import nn
from torch.distributed.pipelining import pipe_split

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels, out_channels, stride: int=1, downsample: Optional[nn.Module] = None,):
        super().__init__()
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.module = nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels, out_channels, stride)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU()),
            ('conv2', conv3x3(out_channels, out_channels)),
            ('bn2', nn.BatchNorm2d(out_channels))
            ]))
        # downsampled_size = output_size(output_size(input_size)) # 2 conv layers
        # self.downsample = nn.AdaptiveAvgPool2d(downsampled_size) # residual connection

    def forward(self, x):
        out = self.module(x)
        identity = self.downsample(x) if self.downsample is not None else x 
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels, out_channels, stride: int=1, downsample: Optional[nn.Module] = None,):
        super().__init__()
        base_width = 64
        width = int(in_channels * (base_width/64.0))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.module = nn.Sequential(OrderedDict([
            ('conv1', conv1x1(in_channels, width)),
            ('bn1', nn.BatchNorm2d(width)),
            ('relu1', self.relu),
            ('conv2', conv3x3(width, width, stride=stride)),
            ('bn2', nn.BatchNorm2d(width)),
            ('relu2', self.relu),
            ('conv3', conv1x1(width, out_channels * self.expansion)),
            ('bn3', nn.BatchNorm2d(out_channels * self.expansion))
            ]))

    def forward(self, x):
        out = self.module(x)
        identity = self.downsample(x) if self.downsample is not None else x 
        return self.relu(out + identity)

class ResNet(nn.Module):
    def __init__(self, num_classes: int, model_size: int=50):
        super().__init__()
        if model_size == 18:
            block, layers = BasicBlock, [2, 2, 2, 2]
        elif model_size == 34:
            block, layers = BasicBlock, [3, 4, 6, 3]
        elif model_size == 50:
            block, layers = Bottleneck, [3, 4, 6, 3]
        elif model_size == 101:
            block, layers = Bottleneck, [3, 4, 23, 3]
        elif model_size == 152:
            block, layers = Bottleneck, [3, 8, 36, 3]
        else:
            raise ValueError(f"model_size should be 18, 34, 50, 101, or 152 but got {model_size}")
        
        self.in_channels = 64
        # self.head = nn.Linear(512 * block.expansion, num_classes)

        self.module = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(self.in_channels)),
            ('relu1', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('layer1', self._make_layer(block, 64, layers[0])),
            ('layer2', self._make_layer(block, 128, layers[1], stride=2)),
            ('layer3', self._make_layer(block, 256, layers[2], stride=2)),
            ('layer4', self._make_layer(block, 512, layers[3], stride=2)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('head', nn.Linear(512 * block.expansion, num_classes))
        ]))
        # self.module = nn.ModuleDict({
        #     'conv1': nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
        #     'bn1': nn.BatchNorm2d(self.in_channels),
        #     'relu1': nn.ReLU(),
        #     'maxpool': nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     'layer1': self._make_layer(block, 64, layers[0]),
        #     'layer2': self._make_layer(block, 128, layers[1], stride=2),
        #     'layer3': self._make_layer(block, 256, layers[2], stride=2),
        #     'layer4': self._make_layer(block, 512, layers[3], stride=2),
        #     'avgpool': nn.AdaptiveAvgPool2d((1, 1)),
        #     'flatten': nn.Flatten(),
        #     'head': self.head
        # })
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        layers += [block(self.in_channels, out_channels) for _ in range(1, num_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)
        # for i, name, layer in enumerate(self.module.items()):
        #     x = layer(x)
        #     if i in [5,6,7]:
        #         pipe_split()

import torch
from torch.distributed.pipelining import PipelineStage
from utils import args
def manual_model_split(module:nn.Sequential, input_shape, balance) -> PipelineStage:
    '''Manually split the model into pipeline stages.'''
    num_stages = args.pp
    current_stage = args.pp_rank # stage of this process
    assert  len(balance) == num_stages, f"balance should have the same length as the number of pipeline stages. balance : {balance}"
    assert  module and isinstance(module, nn.Sequential), "model's layers should be wrapped in self.module in nn.Sequential(OrderedDict)"
    
    num_layers = len(module)
    start_layer = [0] + [sum(balance[:i]) for i in range(1, len(balance))]
    last_layer = [sum(balance[:i+1])-1 for i in range(len(balance))]
    assert num_layers-1 == last_layer[-1], f"balance should sum to the number of layers in the model. num_layers : {num_layers}, balance : {balance}"

    # get the input shape of the current stage
    input_example = torch.randn(input_shape).chunk(args.microbatches)[0]
    for i, layer in enumerate(module):
        if i == start_layer[current_stage]:
            break
        input_example = layer(input_example)
    
    # remove all other stages except the current stage
    for stage in range(num_stages-1, -1, -1):
        for i in range(last_layer[stage], start_layer[stage]-1, -1):
            if stage != current_stage:
                del module[i]

    stage = PipelineStage(module, stage_index=current_stage, num_stages=num_stages, 
                          device=args.local_rank, group=args.pp_group, input_args=input_example)
    return stage

