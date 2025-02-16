import torch
from beartype import beartype

@beartype
def save_checkpoint(state:dict, filepath:str):
    """Save model training checkpoints to local file path

    Parameters
    ----------
    state : dict
        The model state dictionary to write to disk
    filepath : str
        The local file path to write the checkpoints to
    
    Returns
    -------
    """
    torch.save(state, filepath)

@beartype
def load_checkpoint(filepath:str) -> dict:
    """Load model training checkpoints from local file path

    Parameters
    ----------
    filepath : str
        The local file path to read the checkpoints from
    
    Returns
    -------
    """
    # load checkpoint file
    checkpoint = torch.load(filepath, weights_only=False)
    return checkpoint