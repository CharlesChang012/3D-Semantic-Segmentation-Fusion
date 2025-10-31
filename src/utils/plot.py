import matplotlib as plt

def plot_training_history(train_his, val_his):
    x = np.arange(len(train_his))
    plt.figure()
    plt.plot(x, torch.tensor(train_his, device='cpu'))
    plt.plot(x, torch.tensor(val_his, device='cpu'))
    plt.legend(['Training top1 accuracy', 'Validation top1 accuracy'])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Top1 Accuracy')
    plt.title('3DSSF')
    plt.show()