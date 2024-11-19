from torch.cuda import nvtx as cuda_nvtx
from utils.util import args
import nvtx

def mark_step_epoch_nvtx(step, epoch=0):
    '''print the epoch nvtx'''
    cuda_nvtx.mark(message=f"step {step} of epoch {epoch}")
    return

@nvtx.annotate(color="red", domain=f"Rank{args.pp_rank}", category="forward")
def forward_nvtx(microbatch, start=True):
    '''mark the range of forward in nvtx'''
    if start:
        cuda_nvtx.range_push(f"forward_{microbatch}")
    else:
        cuda_nvtx.range_pop()
    return

@nvtx.annotate(color="blue", domain=f"Rank{args.pp_rank}", category="backward")
def backward_nvtx(microbatch, start=True):
    '''mark the range of backward in nvtx'''
    if start:
        cuda_nvtx.range_push(f"backward_{microbatch}")
    else:
        cuda_nvtx.range_pop()

@nvtx.annotate(domain=f"Rank{args.pp_rank}", category="forward")
def forward_recv_nvtx(microbatch, start=True):
    '''mark the range of forward receive ~ send in nvtx'''
    if start:
        cuda_nvtx.range_push(f"recv_to_send_{microbatch}")
    else:
        cuda_nvtx.range_pop()
    return

@nvtx.annotate(domain=f"Rank{args.pp_rank}", category="backward")
def backward_recv_nvtx(microbatch, start=True):
    '''mark the range of backward receive ~ send in nvtx'''
    if start:
        cuda_nvtx.range_push(f"recv_to_send_{microbatch}")
    else:
        cuda_nvtx.range_pop()
    return