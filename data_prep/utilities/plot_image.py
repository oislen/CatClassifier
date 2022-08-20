import matplotlib.pyplot as plt

def plot_image(image_array):
    """"""
    # set plot figure size
    plt.figure(figsize = (8, 6))
    # plot image
    plt.imshow(image_array)
    # show plot
    plt.show()
    return 0
