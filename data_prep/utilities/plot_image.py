import matplotlib.pyplot as plt

def plot_image(image_array, output_fpath = None):
    """"""
    # set plot figure size
    plt.figure(figsize = (8, 6))
    # plot image
    plt.imshow(image_array)
    #  show save, plot and close
    if output_fpath != None:
        plt.savefig(output_fpath)
    # show plot
    plt.show()
    plt.close()
    return 0
