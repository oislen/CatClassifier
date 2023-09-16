import torch

def save(model, output_fpath):
    torch.save(model.state_dict(), output_fpath)
    msg = f'Saved to {output_fpath}'
    return msg