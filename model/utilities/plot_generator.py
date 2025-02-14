import matplotlib.pyplot as plt
from typing import Union
from beartype import beartype

@beartype
def plot_generator(
    generator,
    mode:str='keras',
    output_fpath:Union[str,None]=None,
    show_plot:bool=True
    ):
    """
    Plots multiple images from a generator.

    Parameters
    ----------
    generator : iterator
        The image to plot
    mode : str
        The model mode type the generate was created with, default is 'keras'
    output_fpath : str
        The output file location to save the plot, default is None
    show_plot : bool
        Whether to show the generated plot, default is True

    Returns
    -------
    """
    # plot example
    plt.figure(figsize=(12, 12))
    for i in range(0, 15):
        plt.subplot(5, 3, i+1)
        if mode == 'keras':
            for X_batch, Y_batch in generator:
                image = X_batch[0]
                plt.imshow(image)
                break
        elif mode == 'torch':
            X_batch = generator[i]
            image = X_batch[0]
            plt.imshow(image)
    plt.tight_layout()
    #  show save, plot and close
    if output_fpath != None:
        plt.savefig(output_fpath)
    if show_plot:
        plt.show()
    plt.close()