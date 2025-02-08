import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from typing import Union

@beartype
def plot_image(
    image_array:np.array,
    output_fpath:Union[str,None]=None,
    show_plot:bool=True
    ):
    """
    Plots an image array.

    Parameters
    ----------
    image_array : PIL
        The image to plot
    output_fpath : str
        The output file location to save the plot, default is None
    show_plot : bool
        Whether to show the generated plot, default is True

    Returns
    -------
    """
    # set plot figure size
    plt.figure(figsize = (8, 6))
    # plot image
    plt.imshow(image_array)
    #  show save, plot and close
    if output_fpath != None:
        plt.savefig(output_fpath)
    if show_plot:
        # show plot
        plt.show()
    plt.close()
