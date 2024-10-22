import torch

def load(model, input_fpath, weights_only=False):
    model.load_state_dict(torch.load(input_fpath, weights_only=weights_only))
    model.eval()
    msg = f'Loaded from {input_fpath}'
    return msg