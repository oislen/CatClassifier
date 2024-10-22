import torch
from beartype import beartype

@beartype
def load(model, input_fpath:str, weights_only:bool=False):
    """Loads a torch model from disk as a file
    
    Parameters
    ----------
    input_fpath : str
        The input file location to load the torch model from disk
    weights_only : bool
        Whether loading just the model weights or the full serialised model object, default is False
    
    Returns
    -------
    str
        The load model message status
    """
    model.load_state_dict(torch.load(input_fpath, weights_only=weights_only))
    model.eval()
    msg = f'Loaded from {input_fpath}'
    return msg