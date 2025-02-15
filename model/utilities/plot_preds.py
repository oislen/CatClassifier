import os
from beartype import beartype
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
import cons

@beartype
def plot_preds(
    data:pd.DataFrame,
    output_fpath:str = None,
    show_plot:bool=True
    ):
    """
    Shows model predictions as a grid of images with labels

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe of image arrays and associated labels
    output_fpath : str
        The file path to save the plot of predictions, default is None
    show_plot : bool
        Whether to show the generated plot, default is True

    Returns
    -------
    """
    sample_test = data.head(18)
    plt.figure(figsize=(12, 24))
    for id, (index, row) in enumerate(sample_test.iterrows()):
        filename = row['filenames']
        category = row['category']
        img = load_img(os.path.join(cons.test_fdir, filename), target_size=cons.IMAGE_SIZE)
        plt.subplot(6, 3, id+1)
        plt.imshow(img)
        #plt.title(category)
        plt.xlabel(filename + '(' + "{}".format(category) + ')' )
    plt.tight_layout()
    #  show save, plot and close
    if output_fpath != None:
        plt.savefig(output_fpath)
    if show_plot:
        plt.show()
    plt.close()