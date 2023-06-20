from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np

def calculate_mean_std(raw_data, batch_size=200):
    raw_loader = DataLoader(raw_data, batch_size=batch_size, num_workers=4, shuffle=False)

    num_samples = len(raw_loader.dataset)
    num_batches = num_samples // batch_size

    data_mean = 0.
    data_std = 0.
    for i, (images, _) in enumerate(raw_loader):
        print(f"computing batch {i+1:8} / {num_batches}", end="\r")
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        data_mean += images.mean(2).sum(0)
        data_std += images.std(2).sum(0)

    data_mean /= num_samples
    data_std /= num_samples

    print("")
    print(data_mean, data_std)
    
    return data_mean, data_std

def split_train_val_test(dataset, val=0.1, test=0.1, batch_size=256):

    n = len(dataset)  # total number of examples
    i_val  = int(val * n)  # take ~10% for test
    i_test = int(test * n)  # take ~10% for test

    data_train, data_val, data_test = random_split(dataset, [n-i_val-i_test, i_val, i_test])
    data = {}

    data["train"] = DataLoader(data_train, batch_size=batch_size, num_workers=4, shuffle=True)
    if i_val > 0:
         data["val"]   = DataLoader(data_val,   batch_size=batch_size, num_workers=4, shuffle=True)
    if i_test > 0:
        data["test"]  = DataLoader(data_test,  batch_size=batch_size, num_workers=4, shuffle=True)

    return data

def visualize_N_of_class(n, loader_train):

    for i, (X, y) in enumerate(loader_train):
        print(X.shape)
        print(X.min(), X.max())
        print(X.max() - X.min())

        #data unnormalization! be careful! this is not what NN actually is about to see
        X = (((X - X.min()) / (X.max()-X.min()))*255).type(torch.int)

        classes = [0, 1]
        num_classes = len(classes)
        samples_per_class = n
        for c_i, cls in enumerate(classes):
            idxs = np.flatnonzero(y == c_i)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for j, idx in enumerate(idxs):
                plt_idx = j * num_classes + c_i + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(X[idx].permute(1,2,0))
                plt.axis('off')
                if j == 0:
                    plt.title(cls)
        plt.show()
        break

    return