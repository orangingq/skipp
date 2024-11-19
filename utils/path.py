import os

default_path = {
    "checkpoint": "/data3/shcho/skipp/checkpoints",
    "dataset": "/data3/shcho/datasets",
}

def get_default_path(key:str)->str:
    '''Get the default path of the specified key. 
    Args:
        key (str): the key of the default path. choose one of ["checkpoint", "dataset"]
    '''
    assert key in default_path.keys(), f"key must be one of {default_path.keys()}"
    return default_path[key]

def get_abs_path(path:str, key:str)->str:
    '''Get the absolute path of the specified path. 
    Args:
        path (str): the path to be joined with the default path.
        key (str): the key of the default path. choose one of ["checkpoint", "dataset"]
    '''
    assert key in default_path.keys(), f"key must be one of {default_path.keys()}"
    
    path = os.path.abspath(os.path.join(default_path[key], path))
    return path