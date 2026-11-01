import logging

import torch
from beartype import beartype

# Minimum spatial size required by the pretrained backbones (AlexNet, VGG-16,
# ResNet-50) for the legacy TorchScript ONNX tracer to resolve adaptive_avg_pool2d.
# The training crop size (128×128) is too small; 224×224 is the canonical backbone size.
_ONNX_MIN_SPATIAL = 224


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
        The input image width used during training (recorded in the model metadata)
    image_height : int
        The input image height used during training (recorded in the model metadata)

    Returns
    -------
    str
        The export status message
    """
    model.eval()
    # ONNX export runs on CPU regardless of training device; move model temporarily
    original_device = next(model.parameters()).device
    model.cpu()
    # Use 224×224 for tracing: the pretrained backbones require this minimum spatial
    # size for adaptive_avg_pool2d to be traceable by the legacy ONNX exporter.
    # The exported graph uses a dynamic batch axis so any batch size is accepted
    # at inference time; spatial dimensions are fixed at 224×224.
    trace_width = max(image_width, _ONNX_MIN_SPATIAL)
    trace_height = max(image_height, _ONNX_MIN_SPATIAL)
    dummy_input = torch.randn(1, 3, trace_height, trace_width)
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
    model.to(original_device)
    msg = f"Exported ONNX model to {output_fpath}"
    logging.info(msg)
    return msg
