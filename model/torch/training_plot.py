import matplotlib.pyplot as plt

def training_plot(train_loss_list, train_acc_list):
    """
    """
    num_epochs = len(train_loss_list)
    plt.figure()
    plt.title('Train Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.plot(range(1, num_epochs+1), np.array(train_loss_list), color='blue', linestyle='-', label='Train_Loss')
    plt.plot(range(1, num_epochs+1), np.array(train_acc_list), color='red', linestyle='-', label='Train_Accuracy')
    plt.legend()
    plt.show() 
    return 0