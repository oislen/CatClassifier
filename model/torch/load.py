import torch

def load(model, input_fpath):
    model.load_state_dict(torch.load(input_fpath))
    model.eval()
    msg = f'Loaded from {input_fpath}'
    return msg