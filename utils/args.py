
import argparse
import os
import torch.distributed as dist
import torch

seed = 42
local_rank = 0

# distributed learning
dp = 1
pp = 1
pp_scheduler = 'gpipe'
distributed = False
world_size = 1
dataset = 'cifar-10'
batch_size = 128
microbatches = 4 # number of microbatches for pipeline parallelism
model_name = 'mixer'
checkpoint = None
pretrained = None
optimizer = 'adam'
scheduler = None
learning_rate = 0.001
early_stop = False
num_epochs = 300
warmup_epochs = 5
run_name = None
save_dir = None
log_freq = 100
wandb = None
# step_name = Array(ctypes.c_char, b'   ')
# monitor_message = Array(ctypes.c_char_p, (' '*100).encode())
# stop_event = Event() 
# monitor_process = None

# distributed learning
pp_group, pp_group_ranks = None, [] # pipeline parallelism group & (local ranks in this group) of this process
dp_group, dp_group_ranks = None, [] # data parallelism group & (local ranks in this group) of this process
pp_rank = 0 # pipeline parallelism rank : local rank in the pipeline parallelism group
dp_rank = 0 # data parallelism rank : local rank in the data parallelism group
master_rank = 0 # master rank : a unique rank that manages the whole training processes
master_dp_rank = 0 # master dp rank : a dp group that manages the model updates

def get_args():
    global_variables = [
        'seed', 
        'local_rank', 'dp', 'pp', 'pp_scheduler', 'distributed', 'world_size',
        'dataset', 'batch_size', 'microbatches',
        'model_name', 'checkpoint', 'pretrained', 
        'optimizer', 'scheduler', 'learning_rate', 'early_stop', 'num_epochs', 'warmup_epochs', 
        'run_name', 'save_dir', 'log_freq', 
        'wandb', 'pp_rank', 'dp_rank', 'master_rank', 'master_dp_rank'
    ]
    return {var: globals()[var] for var in global_variables}

def set_args():
    global seed
    global local_rank, dp, pp, pp_scheduler, distributed, world_size
    global dataset, batch_size, microbatches
    global model_name, checkpoint, pretrained
    global optimizer, scheduler, learning_rate, early_stop, num_epochs, warmup_epochs
    global run_name, save_dir, log_freq
    global wandb, pp_rank, dp_rank, master_rank, master_dp_rank
    
    parser = argparse.ArgumentParser()
    # Arguments
    parser.add_argument('--seed', type=int, default=seed, help='Random Seed')
    # Distributed learning arguments
    # parser.add_argument('--local_rank', type=int, default=local_rank, help='Local Rank. Do not set manually.')
    parser.add_argument('--dp', type=int, default=dp, help='Data Parallelism')
    parser.add_argument('--pp', type=int, default=pp, help='Pipeline Parallelism')
    parser.add_argument('--pp_scheduler', type=str, default=pp_scheduler, choices=['gpipe', '1F1B'], help='Pipeline Parallelism Scheduler')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch Size')
    parser.add_argument('--microbatches', type=int, default=microbatches, help='Number of Microbatches for Pipeline Parallelism')
    # model arguments
    parser.add_argument('--model_name', type=str, default=model_name, help='Model Name')
    parser.add_argument('--checkpoint', type=str, default=checkpoint, help='Checkpoint Load directory')
    parser.add_argument('--pretrained', action='store_true', help='Load Pretrained weights')
    # training arguments
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'sam', 'gsam'], default=optimizer, help='Optimizer')
    parser.add_argument('--scheduler', type=str, choices=[None, 'linear', 'cosine', 'sam', 'gsam'], default=scheduler, help='Scheduler')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning Rate')
    parser.add_argument('--early_stop', action='store_true', help='Stop training early')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='Number of Epochs')
    parser.add_argument('--warmup_epochs', type=int, default=warmup_epochs, help='Warmup Epochs')
    # logging arguments
    parser.add_argument('--run_name', type=str, default=run_name, help='Run Name for wandb. None for not using wandb.')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='Save directory')
    parser.add_argument('--log_freq', type=int, default=log_freq, help='Log Frequency for training')
    args = parser.parse_args()
    for arg in vars(args):
        globals()[arg] = getattr(args, arg)# update global variables
    
    if dp > 1 or pp > 1:
        distributed = True
        dist_setup()
    wandb = (run_name is not None) & (local_rank == master_rank)

    # Print Arguments
    if local_rank == master_rank: 
        var_list = [f'{key}={val}' for key,val in get_args().items()]
        print(f"\n----- Arguments:\n{', '.join(var_list)}\n")
    return


def dist_setup()->int:
    '''
    Initialize distributed learning environment.
    (local_rank, dp_rank, pp_rank) topology example : 
        - dp=2, pp=3 : 
                ----------model_forward------------->
            DP 0  || (0,0,0) | (1,0,1) | (2,0,2) || --> output 0
                    -----------------------------
            DP 1  || (3,1,0) | (4,1,1) | (5,1,2) || --> output 1
                    -----------------------------
                       PP 0      PP 1      PP 2
            master rank : 0 / master dp rank : 0 (=local rank 0,3) / master pp rank : 2 (=local rank 2,5)

    '''
    global dp, pp, distributed, log_freq
    
    local_rank = 0
    if distributed:
        # TODO : currently, only support for a single node (global rank == local rank)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        assert world_size == dp * pp, f"world_size({world_size}) != dp({dp}) * pp({pp})"
        local_rank = int(os.environ["LOCAL_RANK"])
        all_ranks = [i for i in range(world_size)]
        topology = torch.tensor(all_ranks).reshape(dp, pp)
        dp_rank, pp_rank = [int(i.item()) for i in torch.where(topology == local_rank)]
        dp_group_ranks = topology[:, pp_rank].tolist() # ranks in the same data parallelism group
        pp_group_ranks = topology[dp_rank].tolist()

        assert len(dp_group_ranks) == dp, f"dp_group should have {dp} ranks. dp_group_ranks : {dp_group_ranks}"
        assert len(pp_group_ranks) == pp, f"pp_group should have {pp} ranks. pp_group_ranks : {pp_group_ranks}"
        
        # update global variables
        globals()['local_rank'] = local_rank
        globals()['world_size'] = world_size
        globals()['dp_rank'] = dp_rank # local_rank // pp
        globals()['pp_rank'] = pp_rank  # local_rank % pp
        globals()['master_rank'] = 0
        globals()['master_dp_rank'] = 0
        globals()['pp_group_ranks'] = pp_group_ranks
        globals()['dp_group_ranks'] = dp_group_ranks
        globals()['pp_group'] = dist.new_group(ranks=pp_group_ranks, backend='nccl')
        globals()['dp_group'] = dist.new_group(ranks=dp_group_ranks, backend='nccl')
        globals()['log_freq'] = log_freq if (log_freq and local_rank == (dp_rank*pp+(pp-1))) else None
        
        dist.barrier(device_ids=[local_rank])

        return local_rank

