import logging

import onnxruntime as ort
from beartype import beartype


@beartype
def import_onnx(input_fpath: str, providers: list = None) -> ort.InferenceSession:
    """Loads an ONNX model from disk as an ONNX Runtime InferenceSession.

    Parameters
    ----------
    input_fpath : str
        The input file location to load the ONNX model from disk (e.g., model.onnx)
    providers : list, optional
        Execution providers for ONNX Runtime (default is ['CPUExecutionProvider'])

    Returns
    -------
    ort.InferenceSession
        The initialized ONNX runtime session ready for inference
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    # Initialize the ONNX session using the file path directly
    session = ort.InferenceSession(input_fpath, providers=providers)

    msg = f"Loaded ONNX model from {input_fpath} using providers {providers}"
    logging.info(msg)

    return session
