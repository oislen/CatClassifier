class ModelFit():
    def __init__(self, loss, accuracy, val_loss, val_accuracy):
        history = {'loss':loss, 'accuracy':accuracy, 'val_loss':val_loss, 'val_accuracy':val_accuracy}
        self.history = history