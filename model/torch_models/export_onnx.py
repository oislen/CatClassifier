import logging

import torch
from beartype import beartype


@beartype
def export_onnx(model, output_fpath: str, image_width: int, image_height: int) -> str:
    """Exports a torch model to ONNX format.

    Parameters
    ----------
    model : CustomModelClass
        The trained torch model to export
    output_fpath : str
        The output file location to write the ONNX model to disk
    image_width : int
        The input image width expected by the model
    image_height : int
        The input image height expected by the model

    Returns
    -------
    str
        The export status message
    """
    model.eval()
    dummy_input = torch.randn(1, 3, image_width, image_height)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_fpath,
            export_params=True,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
    msg = f"Exported ONNX model to {output_fpath}"
    logging.info(msg)
    return msg
