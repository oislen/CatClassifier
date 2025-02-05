import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

def plot_preds(data, cons, output_fpath = None):
    """"""
    sample_test = data.head(18)
    plt.figure(figsize=(12, 24))
    for id, (index, row) in enumerate(sample_test.iterrows()):
        filename = row['filename']
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
    plt.show()
    plt.close()
    return 0