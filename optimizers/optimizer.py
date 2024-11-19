import math
import torch.nn as nn
import torch
import torch.optim as optim
from utils import args
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer_scheduler(model:nn.Module) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler, bool]:
    '''
    Get optimizer based on the optimizer name.
    optimizer : str : name of the optimizer
    model : nn.Module : model to optimize
    '''
    closure = False

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=5e-5)
        scheduler = get_scheduler(optimizer)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-5)
        scheduler = get_scheduler(optimizer)
    else :
        raise ValueError(f"Optimizer {optimizer} is not supported.")
    
    return optimizer, scheduler, closure


def get_scheduler(optimizer:optim.Optimizer) -> optim.lr_scheduler:
    if args.scheduler is None:
        return None
    assert args.scheduler in ['linear', 'cosine'], f"Scheduler {args.scheduler} is not supported."
    
    # Define a learning rate scheduler with linear warmup and linear decay
    if args.scheduler == 'linear':
        def lr_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    # Define a learning rate scheduler with cosine warmup and cosine decay
    elif args.scheduler == 'cosine':
        min_lr = 1e-3 # min(1e-6, args.learning_rate*0.1)/args.learning_rate
        def lr_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int) -> float:
            if args.early_stop:
                num_training_steps = min(100, num_training_steps)
            if current_step < num_warmup_steps:
                return float(current_step) / float(num_warmup_steps)
            else:
                return (min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps))))
        
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, num_warmup_steps=max(1, args.warmup_epochs), num_training_steps=args.num_epochs))
    
    return scheduler

# https://github.com/pytorch/torchtitan/blob/main/torchtitan/optimizer.py#L15
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import functools
import torch

# # consider split between PP and non-PP
# def build_optimizers(model_parts):
#     """Wrap one optimizer per model part in an OptimizersContainer which provides a single
#     step() and zero_grad() method for all the child optimizers.
#     """

#     def _build_optimizer(model):
#         name = job_config.optimizer.name
#         lr = job_config.optimizer.lr
#         fused = job_config.optimizer.fused

#         # Common parameters for both optimizers
#         optimizer_kwargs = {
#             "lr": lr,
#             "betas": (0.9, 0.95),
#             "weight_decay": 0.1,
#             "fused": fused,
#             "foreach": not fused,
#         }
#         if name == "Adam":
#             # TODO: make the optimizer options configurable by toml/cmd args
#             optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
#         elif name == "AdamW":
#             optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
#         else:
#             raise NotImplementedError(f"Optimizer {name} not added.")

#         return optimizer

#     class OptimizersContainer:
#         """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

#         def __init__(self, optimizers):
#             self.optimizers = optimizers

#         def step(self):
#             for optimizer in self.optimizers:
#                 optimizer.step()

#         def zero_grad(self):
#             for optimizer in self.optimizers:
#                 optimizer.zero_grad()

#     return OptimizersContainer([_build_optimizer(model) for model in model_parts])



# def build_lr_schedulers(optimizers, job_config: JobConfig):
#     def _build_lr_scheduler(optimizer):
#         """Build a linear warmup and linear decay scheduler"""
#         warmup_steps = int(job_config.training.warmup_steps)
#         decay_steps = float(max(1, job_config.training.steps - warmup_steps))
#         lr_lambda = functools.partial(
#             linear_warmup_linear_decay, warmup_steps, decay_steps
#         )
#         warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
#         return warmup_scheduler

#     class SchedulersContainer:
#         """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

#         def __init__(self, schedulers):
#             self.schedulers = schedulers

#         def step(self):
#             for schedulers in self.schedulers:
#                 schedulers.step()

#     return SchedulersContainer(
#         [_build_lr_scheduler(optimizer) for optimizer in optimizers]
#     )