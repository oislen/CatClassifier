import matplotlib.pyplot as plt

def plot_generator(generator, mode = 'keras', output_fpath = None):
    """"""
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
            X_batch, Y_batch = generator[i]
            image = X_batch[0]
            plt.imshow(image)
    plt.tight_layout()
    #  show save, plot and close
    if output_fpath != None:
        plt.savefig(output_fpath)
    plt.show()
    plt.close()
    return 0