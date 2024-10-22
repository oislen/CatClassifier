import torch
from beartype import beartype

@beartype
def save(model, output_fpath:str) -> str:
    """
    Writes a torch model to disk as a file
    
    Parameters
    ----------
    model : CustomModelClass
        The customer torch model class object being fit
    output_fpath : str
        The output file location to write the torch model to disk
    
    Returns
    -------
    str
        The model data message status
    """
    torch.save(model.state_dict(), output_fpath)
    msg = f'Saved to {output_fpath}'
    return msg