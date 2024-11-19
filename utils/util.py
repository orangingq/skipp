from datetime import datetime
import timeit
import os
import numpy as np
import torch
import random
from . import args
from .nvtx import forward_nvtx, backward_nvtx, forward_recv_nvtx, backward_recv_nvtx

def random_seed(seed):
    '''set random seed'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_first_stage_rank():
    '''get the first stage rank in the pipeline parallelism group.'''
    return args.dp_rank * args.pp 

def get_last_stage_rank():
    '''get the last stage rank in the pipeline parallelism group.'''
    return args.dp_rank * args.pp + (args.pp - 1)

def log_time(message='', rank=True, timestamp=None, end='\n'):
    '''log the time.'''
    if timestamp is None:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-2]
    rank = f"[RANK{args.local_rank}]" if rank else ''
    message = f"{timestamp} {rank} {message}{end}"
    print(message, end='', flush=True)
    return message

last_time = timeit.default_timer()
def pipeline_log(microbatch=0, step='', postfix='', timestamp=None, log=True, type=0):
    assert type in [0, 1], "type should be 0 or 1"
    '''log the pipeline step.'''
    # pipeline steps
    start = ['start']
    forward = ['fwd_from', 'fwd_to'] # ['fwd_recv', 'fwd', 'fwd_send']
    backward = ['bwd_from', 'bwd_to'] # ['bwd_recv', 'bwd', 'bwd_send']
    end = ['update', 'finish']

    # set nvtx 
    if step in ['fwd_from', 'fwd_to']:
        forward_nvtx(microbatch, start=step.endswith('from'))
    elif step in ['bwd_from', 'bwd_to']:
        backward_nvtx(microbatch, start=step.endswith('from'))
    elif step in ['fwd_recv', 'fwd_send']:
        forward_recv_nvtx(microbatch, start=step.endswith('recv'))
    elif step in ['bwd_recv', 'bwd_send']:
        backward_recv_nvtx(microbatch, start=step.endswith('recv'))
    # if step == 'start':
    #     nvtx.range_push(f"start_rank{args.pp_rank}")
    # elif step == 'fwd_from':
    #     nvtx.range_push(f"forward_{microbatch}_rank{args.pp_rank}")
    # elif step == 'fwd_recv':
    #     nvtx.range_push(f"fwd_recv_{microbatch}_rank{args.pp_rank}")
    # elif step == 'bwd_from':
    #     nvtx.range_push(f"backward_{microbatch}_rank{args.pp_rank}")
    # elif step == 'bwd_recv':
    #     nvtx.range_push(f"bwd_recv_{microbatch}_rank{args.pp_rank}")
    # elif step in ['bwd_to', 'bwd_send', 'fwd_to', 'fwd_send', 'finish']:
    #     nvtx.range_pop()

    if type == 0: # print per rank
        if step in start and log and args.pp_rank == 0:
            ranks = [f'GPU{k}'.center((len(str(args.microbatches))+3)) for k in range(args.pp)]
            log_time(f" {'|'.join(ranks)}  - {step:9} {postfix}", rank=False, timestamp=timestamp)
            return step, step
        elif step in forward:
            step_name = 'F' + str(microbatch)
        elif step in backward:
            step_name = 'B' + str(microbatch)
        elif step in end:
            step_name = 'E' + str(microbatch)
            if step == 'finish':
                ranks = ['#'*(len(str(args.microbatches))+3)]*args.pp
                log_time(f" {'#'.join(ranks)}  - {step:9} {postfix}", rank=False, timestamp=timestamp, end='\n\n')
                return step, step
        else:
            return None, None
        
        # add time duration
        now_time = timeit.default_timer()
        global last_time
        if step.endswith('to') or step == 'finish':
            postfix = f"{now_time - last_time:.4f}s " + postfix
        last_time = now_time

        # log the step
        if log:
            ranks = ['-'*(len(str(args.microbatches))+1)]*args.pp
            ranks[args.pp_rank] = step_name
            log_time(f"[ {' | '.join(ranks)} ] - {step:9} {postfix}", rank=False, timestamp=timestamp)
        return step_name, step

    elif type == 1: # print per microbatch
        final_step = len(forward)* args.pp + len(backward)*args.pp + len(end)
        if step in start and log and args.pp_rank == 0:
            batches = [f'MB{k}'.center((len(str(final_step))+2)) for k in range(args.microbatches)]
            log_time(f" {'|'.join(batches)}  - {step:9} {postfix}", timestamp=timestamp)
            return step, step
        elif step in forward:
            step_num = forward.index(step) + len(forward)*args.pp_rank
        elif step in backward:
            step_num = backward.index(step) + len(forward)* args.pp + len(backward)*(args.pp-1 - args.pp_rank)
        elif step in end:
            step_num = end.index(step) + len(forward)* args.pp + len(backward)*args.pp
            if step == 'finish':
                batches = ['#'*(len(str(final_step))+2)]*args.microbatches
                log_time(f" {'#'.join(batches)}  - {step:9} {postfix}", timestamp=timestamp, end='\n\n')
                return step, step
        else:
            return None, None

        if log:
            batches = ['-'*len(str(final_step))]*args.microbatches
            if 0 <= microbatch and microbatch < args.microbatches:
                batches[microbatch] = f"{step_num:{len(str(final_step))}d}"
            else:
                batches = ['#'*len(str(final_step))]*args.microbatches
            if len(postfix) > 0:
                postfix = ': ' + postfix
            if step == 'fwd' : step = 'forward '
            elif step == 'bwd' : step = 'backward'
            log_time(f"[ {' | '.join(batches)} ] - {step:9} {postfix}", timestamp=timestamp)
        return step_num, step
