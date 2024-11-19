from typing import Iterator
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipelining import PipelineStage, pipeline
from torch.nn.parameter import Parameter

from models.ResNet import ResNet, manual_model_split
from utils import args
from utils.util import pipeline_log


class ModelWrapper(nn.Module):
    __slots__ = ['model', 'type']
    '''Model Wrapper to handle PipelineStage and DDP models.'''
    def __init__(self, model):
        super().__init__()
        assert isinstance(model, (nn.Module, PipelineStage, DDP)), f"model should be nn.Module, PipelineStage, or DDP. Got {model}"
        self.model = model
        if isinstance(model, PipelineStage):
            self.model.submod.register_forward_pre_hook(self._forward_pre_hook)
            self.model.submod.register_forward_hook(self._forward_hook)
            self.model.submod.register_full_backward_pre_hook(self._backward_pre_hook)
            self.model.submod.register_full_backward_hook(self._backward_hook)
        self.type = type(model) # nn.Module, PipelineStage, DDP
        self.microbatches_forward = [None] * args.microbatches
        self.microbatches_backward = [None] * args.microbatches
        self.log_pp = True
        return
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.type == PipelineStage:
            for layer in self.model.submod:
                for name, param in layer.named_parameters(recurse=recurse):
                    yield param
        elif self.type == DDP:
            return self.model.module.parameters(recurse)
        return super().parameters(recurse)
    
    def _forward_pre_hook(self, module, input):
        if self.log_pp:
            input_val = input[0].std().item()
            try:
                microbatch = self.microbatches_forward.index(None)
            except ValueError:
                print(f"microbatches_forward : {self.microbatches_forward}")
                microbatch = -1
            self.microbatches_forward[microbatch] = input_val
            pipeline_log(microbatch, step='fwd_from', postfix=f"forward start") 
        return

    def _forward_hook(self, module, input, output):
        if self.log_pp:
            input_val, output_val = input[0].std().item(), output.std().item()
            microbatch = self.microbatches_forward.index(input_val)
            self.microbatches_forward[microbatch] = output_val
            pipeline_log(microbatch, step='fwd_to', postfix=f"forward end")
        return
    
    def _backward_pre_hook(self, module, input):
        if self.log_pp:
            grad_input_val = input[0].std().item()
            try:
                microbatch = self.microbatches_backward.index(None)
            except ValueError:
                print(f"microbatches_backward : {self.microbatches_backward}")
                microbatch = -1
            self.microbatches_backward[microbatch] = grad_input_val
            pipeline_log(microbatch, step='bwd_from', postfix=f"backward start") 
        return

    def _backward_hook(self, module, grad_output, grad_input):
        if self.log_pp:
            grad_input_val = grad_input[0].std().item()
            microbatch = self.microbatches_backward.index(grad_input_val)
            if args.pp_rank > 0:
                grad_output_val = grad_output[0].std().item()
                self.microbatches_backward[microbatch] = grad_output_val
                pipeline_log(microbatch, step='bwd_to', postfix=f"backward end") #  (grad_input.std={grad_input_val:.2f}, grad_output.std={grad_output_val:.2f})
            else:
                grad_output_val = None
                pipeline_log(microbatch, step='bwd_to', postfix=f"backward end") #  (grad_input.std={grad_input_val:.2f})
        return
    
    def __getattr__(self, name):
        if name in self.__slots__:
            return super().__getattr__(name)
        return getattr(self.model, name)
    
    def __setattr__(self, name, value):
        if name in self.__slots__:
            return super().__setattr__(name, value)
        return setattr(self.model, name, value)

    def __delattr__(self, name):
        if name in self.__slots__:
            return super().__delattr__(name)
        return delattr(self.model, name)
    
    def __str__(self):
        return str(self.model)
    
    def get_log_pp(self):
        return self.log_pp

    def set_log_pp(self, log_pp:bool):
        self.log_pp = log_pp
        return

    def initialize(self):
        self.microbatches_forward = [None] * args.microbatches
        self.microbatches_backward = [None] * args.microbatches
        return

def get_model(model:str, batch_shape) -> ModelWrapper:
    model = model.lower()
    if model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']: # 18, 34, 50, 101, or 152
        model = ResNet(
            num_classes=args.num_classes, 
            model_size=int(model.split('resnet')[-1]),
        )
        # model = torch_models.__dict__[model](num_classes=args.num_classes)
    else:
        raise ValueError(f"Model {model} is not supported.")

    # print model parameters
    if args.local_rank == args.master_rank: get_number_of_params(model, prefix="\nTotal Model", verbose=True)
    
    # Model Distribution Setup
    model = dist_model_setup(model, batch_shape)

    return model


def get_number_of_params(model:ModelWrapper, prefix='', verbose=False):
    num_params = 0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{prefix} : {num_params / 10 ** 6 :.2f}M params\n", end='')
    return num_params


def dist_model_setup(model:nn.Module, batch_shape) -> ModelWrapper:
    if args.distributed: # Multi-GPU
        if args.pp > 1: # Pipeline Parallelism

            if model.__class__.__name__ == 'ResNet':
                # from models.torchgpipe_balance import balance_by_time, balance_by_size
                # assert model.module and isinstance(model.module, nn.Sequential), "model's layers should be wrapped in self.module in nn.Sequential(OrderedDict)"
                # param_scale = 4 if args.optimizer == 'adam' else 3
                # balance = balance_by_time(partitions=args.pp, module=model.module, sample=torch.randn(batch_shape))
                # balance = balance_by_size(partitions=args.pp, module=model.module, input=torch.randn(batch_shape), chunks=args.microbatches, param_scale=param_scale)
                balance = [6,1,1,3]
                assert sum(balance) == len(model.module), f"balance should sum to the number of layers in the model. num_layers : {len(model.module)}, balance : {balance}"
                stage = manual_model_split(model.module, batch_shape, balance=balance)
            else:
                raise ValueError(f"Model {model} is not supported for pipeline parallelism.")

            if args.dp > 1: # Data Parallelism
                stage = DDP(stage)
            model = stage
        else: # Only Data Parallelism
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else: # Single GPU
        model = model.to(device=args.local_rank) 
    
    model = ModelWrapper(model)
    
    # print model parameters for each pipeline stage
    if args.pp > 1 and args.dp_rank == args.master_dp_rank:
        get_number_of_params(model, prefix=f"Pipeline stage {args.local_rank} [{balance[args.pp_rank]} layers]", verbose=True)

    dist.barrier(group=args.pp_group, device_ids=[args.local_rank])
    return model

