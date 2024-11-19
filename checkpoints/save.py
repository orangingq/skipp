import shutil
import torch
import torch.nn as nn
import os
import json
import re 
from collections import OrderedDict

from utils import args
from utils.path import get_abs_path

def load_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        load_dir: str):
    '''
    Every process loads the checkpoint from the specified directory.
    Every process should access to this function due to 'comm.barrier()'. 
    '''
    load_dir = get_abs_path(load_dir, 'checkpoint') # if load_dir is not absolute path, make it absolute path
    load_path = load_dir + '/model_args.txt'
    
    if args.local_rank == args.master_rank:
        load_dir = _merge_distributed_checkpoints(load_dir)

    if not (os.path.exists(load_dir) and os.path.exists(load_path)):
        print(f"[RANK{args.local_rank}] The checkpoint does not exists : {load_path}\nStart from the scratch!\n")
        return model, optimizer, 0

    model_args = __make_model_args(model)
    same = __check_model_args(model_args, load_path)
    
    if not same:
        print(f"[RANK{args.local_rank}] The model arguments are different from the saved model arguments. \nStart from the scratch!\n")
        return model, optimizer, 0
    else: 
        last_save = os.path.join(load_dir, "last_save.txt")
        with open(last_save, 'r') as args_file:
            load_file = args_file.readline().rstrip('\n')
            load_file = os.path.join(load_dir, load_file)
            last_epoch = int(re.search(r'epoch_(\d+)', load_file).group(1))
        print(f"[RANK{args.local_rank}] Load checkpoint (epoch:{last_epoch}) from : {load_file}\n")

        map_location = {'cuda:0': 'cuda:%d' % args.local_rank,
                        'cuda:1': 'cuda:%d' % args.local_rank,
                        'cuda:2': 'cuda:%d' % args.local_rank,
                        'cuda:3': 'cuda:%d' % args.local_rank}

        state_dicts = torch.load(load_file, map_location=map_location, weights_only=True)
        if 'optim' in state_dicts:
            model.load_state_dict(state_dicts['model'])
            optimizer.load_state_dict(state_dicts['optim'])
        else:
            model.load_state_dict(state_dicts)
        return model, optimizer, last_epoch

def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer, 
        epoch: int, 
        save_dir: str,
        logs: dict = {},
        first_save: bool = False):
    '''Save the model checkpoint to the specified directory.'''
    if args.dp_rank != args.master_dp_rank:
        print("Only master rank will save the model.")
        return
    
    save_dir = get_abs_path(save_dir, 'checkpoint') # if load_dir is not absolute path, make it absolute path
    if first_save : # if it is the first save
        if args.local_rank == args.master_rank:
            _merge_distributed_checkpoints(save_dir) # merge the partitioned checkpoints if they exists
            # create the directory if it does not exist
            if not os.path.exists(save_dir):
                print(f"Creating directory: {save_dir}")
                os.makedirs(save_dir) 
        
        # save the model arguments
        model_args = __make_model_args(model)
        save_dir = __write_model_args(save_dir, model_args)

    if not first_save and not os.path.exists(save_dir):
        raise ValueError(f"The directory does not exist: {save_dir}")
    
    # save the model
    save_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
    state_dicts = {'model': model.state_dict()}
    if args.local_rank == args.master_rank:
        state_dicts['optim'] = optimizer.state_dict()
    torch.save(state_dicts, save_path)
    last_save = os.path.join(save_dir, "last_save.txt")
    if not os.path.exists(last_save):
        lines = [f"epoch_{epoch}.pt\n", f'Epoch_{epoch}: {logs}\n']
    else:
        with open(last_save) as args_file:
            lines = args_file.readlines()
            if lines:
                lines[0] = f"epoch_{epoch}.pt\n"
            else:
                lines = [f"epoch_{epoch}.pt\n"]
        lines += [f'Epoch_{epoch}: {logs}\n']
    with open(last_save, 'w') as args_file:
        args_file.writelines(lines)
        
    print(f"Model saved at : {save_path}")
    return save_dir


def __make_model_args(model):
    '''
    Make the model arguments as a dictionary.
    '''
    if isinstance(model, nn.Module):
        model_args = vars(model).copy()
    else:
        model_args = model
    
    assert isinstance(model_args, (dict, OrderedDict)), "model_args should be a nn.Module or a dictionary / set."
    
    pop_condition = lambda value: (not isinstance(value, (nn.Module, OrderedDict, dict, str, int, float, bool))) \
                                    or (isinstance(value, (OrderedDict, dict)) and len(value) == 0)
    pop_list = ['_parameters', 'training', 'micro_offset', 'global_rank', 'world_size', 'local_rank', 'num_stages', 'stage_id', 'fwd_map', '_local_stop', 'checkpoint_parallel_write_pipeline']
    pop_list += [key for key, value in model_args.items() if pop_condition(value)]
    remaining_keys = list(set(model_args.keys()) - set(pop_list)) 
    remaining = {key:model_args[key] for key in remaining_keys}

    for key, value in remaining.items():
        # if the value is complex data type, make the model arguments recursively
        if isinstance(value, (nn.Module, OrderedDict, dict)):
            remaining[key] = __make_model_args(value)
        else:
            continue

    return remaining


def __write_model_args(save_dir:str, model_args: dict)->str:
    '''
    Check if the model arguments are the same as the saved model arguments.
    If the arguments are different, make a new directory and model_args.txt file.
    '''
    assert save_dir is not None, "args_path should be provided."
    args_path = os.path.join(save_dir, "model_args.txt")

    if not os.path.exists(args_path):
        with open(args_path, 'w') as args_file:
            json_format = json.dumps(model_args, indent=4)
            print(json_format, file=args_file)
            print(f"Model arguments saved to: {save_dir}/model_args.txt")
        return save_dir

    else :
        same = __check_model_args(model_args, args_path)
        if same:
            return save_dir
        else : # if the arguments are different, make a new directory and model_args.txt file. 
            dirname = os.path.dirname(args_path)
            match = re.search(r'_(\d+)', dirname) # check if the directory name has a number at the end
            if match:
                number = int(match.group()[1:])
                new_dirname = re.sub(r'\d+', str(number+1), dirname)
            else : 
                new_dirname = dirname + '_0'
            if not os.path.exists(new_dirname):
                os.makedirs(new_dirname)
            return __write_model_args(new_dirname, model_args)
        
        

def __check_model_args(model_args:dict, args_path:str=None, saved_args:dict=None, pre='')->bool:
    '''Check if the model arguments are the same as the saved model arguments.'''
    assert saved_args is not None or os.path.exists(args_path), f"args_path does not exist: {args_path}"
    if args_path is not None:
        with open(args_path, 'r') as args_file:
            saved_args = json.load(args_file)
    
    assert saved_args is not None, "saved_args is None."

    if args.pp > 1 and pre == '[_modules]': # loosely checked for the '_modules' key in pipeline parallelism case
        if not (len(set(model_args.keys()) - set(saved_args.keys())) == 0 \
            and all(s.isdigit() for s in (set(saved_args.keys()) - set(model_args.keys())))):
            print(f"Key Check : model_args{pre}.keys(): {model_args.keys()}, saved_args{pre}.keys(): {saved_args.keys()}")
            return
    elif model_args.keys() != saved_args.keys():
        print(f"Key Check : model_args{pre}.keys(): {model_args.keys()}, saved_args{pre}.keys(): {saved_args.keys()}")
        return False
    for key in model_args.keys():
        if type(model_args[key]) != type(saved_args[key]):
            print(f"Type Check : model_args{pre}[{key}]: {type(model_args[key])}, saved_args{pre}[{key}]: {type(saved_args[key])}")
            return False
        if isinstance(model_args[key], (dict, OrderedDict, set)):
            if not __check_model_args(model_args[key], saved_args=saved_args[key], pre=f"{pre}[{key}]"):
                return False
        elif model_args[key] != saved_args[key]:
            print(f"Value Check : model_args{pre}[{key}]: {str(model_args[key])[:30]+'...' if len(str(model_args[key]))>30 else model_args[key]}, saved_args{pre}[{key}]: {str(saved_args[key])[:30]+'...' if len(str(saved_args[key]))>30 else saved_args[key]}")
            return False
    return True

def _merge_distributed_checkpoints(load_dir:str)->str:
    '''Merge the partitioned checkpoints if they exists. Only master rank will merge the checkpoints.'''
    if not is_folder_distributed(load_dir) or args.local_rank != args.master_rank:
        return load_dir
    dir_list = [os.path.join(load_dir, dir) for dir in os.listdir(load_dir)]
    pp = len(dir_list)
    assert dir_list == [f'{load_dir}/pp_{i}' for i in range(pp)], f"dir_list: {dir_list} should be {[load_dir+'/pp_'+str(i) for i in range(pp)]}"
    
    # model_args.txt
    model_args = None
    for i in range(pp):
        with open(f'{load_dir}/pp_{i}/model_args.txt', 'r') as f:
            partitioned_args = json.load(f)
        if model_args is None:
            model_args = {'_modules':{}}
            model_args = partitioned_args
        else:
            for j in partitioned_args['_modules']:
                if j.isdigit():
                    model_args['_modules'][j] = partitioned_args['_modules'][j]
    load_dir = __write_model_args(load_dir, model_args)
    
    # last_save.txt
    last_save = os.path.join(load_dir, f'pp_{pp-1}', "last_save.txt")
    copy_file(last_save, load_dir+"/last_save.txt")

    # epoch_*.pt
    for file in os.listdir(f'{load_dir}/pp_{pp-1}'):
        if file.startswith('epoch_'):
            state_dict = torch.load(f'{load_dir}/pp_{pp-1}/{file}', weights_only=True)
            for i in range(pp-1):
                state_dict['model'].update(torch.load(f'{load_dir}/pp_{i}/{file}', weights_only=True)['model'])
            torch.save(state_dict, f'{load_dir}/{file}')

    print(f"Merged the model arguments from {pp} partitions.\n", end="")
    return load_dir

    
def copy_file(src, dst):
    try:
        shutil.copyfile(src, dst)
    except Exception as e:
        print(f"Error copying file: {e}")

def is_folder_distributed(folder:str):
    return os.path.exists(os.path.join(folder, "pp_0"))