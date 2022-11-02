from preprocessing import data_loader
from model import CNN
from baseline import Baseline
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def train(net, batch_size=32, lr=0.001, num_epochs=30):
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(0)

    if torch.cuda.is_available():
        net = net.cuda()

    train_loader, val_loader = data_loader(batch_size=batch_size)
    
    # cross entropy loss function and adaptive moment estimation optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr = lr, weight_decay=0.1)

    # softmax for predictions
    softmax = nn.Softmax(dim = 1)
    
    # initialize error and loss history
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0
        train_iters = 0

        total_val_loss = 0.0
        total_val_err = 0.0
        val_iters = 0
        
        train_batches = 0
        net.train()
        for batch in train_loader:
            train_batches += 1
            imgs, labels = batch.values()
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = softmax(outputs)
            # find error and loss for training data
            total_train_err += (np.argmax(pred.detach().cpu(), 1) != np.argmax(labels.cpu(), 1)).sum().item()
            total_train_loss += loss.item()
            train_iters += len(labels)

        val_batches = 0
        net.eval()
        for batch in val_loader:
            val_batches += 1
            imgs, labels = batch.values()
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            outputs = net(imgs)
            loss = criterion(outputs, labels)

            pred = softmax(outputs)

            # find error and loss for training data
            total_val_err += (np.argmax(pred.detach().cpu(), 1) != np.argmax(labels.cpu(), 1)).sum().item()
            total_val_loss += loss.item()
            val_iters += len(labels)

        # record the average error (per iteration) and loss (per batch) for each epoch
        train_err[epoch] = total_train_err / train_iters
        train_loss[epoch] = total_train_loss / train_batches
        val_err[epoch] = total_val_err / val_iters
        val_loss[epoch] = total_val_loss / val_batches
        print(f"Epoch {epoch}: Train err: {train_err[epoch]} Val err: {val_err[epoch]} Train loss: {train_loss[epoch]} Val loss: {val_loss[epoch]}")
        # save model
        model_path = "/models/bs{}_lr{}_epoch{}".format(batch_size,
                                              lr,
                                              epoch)
        torch.save(net.state_dict(), model_path)
    return train_err, train_loss, val_err, val_loss

def plot(train_err, train_loss, val_err, val_loss):
    n = len(train_err) # number of epochs

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Train vs Validation Error")
    ax1.plot(range(1,n+1), train_err, label="Train")
    ax1.plot(range(1,n+1), val_err, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error")
    ax1.legend(loc='best')
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_title("Train vs Validation Loss")
    ax2.plot(range(1,n+1), train_loss, label="Train")
    ax2.plot(range(1,n+1), val_loss, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc='best')
    ax2.xaxis.get_major_locator().set_params(integer=True)
    plt.show()

def performance_per_class(net):
    net.eval()
    _, val_loader = data_loader(batch_size=1)
    errors = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0,
        10:0,
        11:0,
        12:0,
        13:0,
        14:0,
    }
    total = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0,
        10:0,
        11:0,
        12:0,
        13:0,
        14:0,
    }
    wrong_guesses = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0,
        10:0,
        11:0,
        12:0,
        13:0,
        14:0,
    }
    guesses = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0,
        10:0,
        11:0,
        12:0,
        13:0,
        14:0,
    }
    softmax = nn.Softmax(dim = 1)
    for batch in val_loader:
        img, label = batch.values()
        output = softmax(net(img))
        pred = np.argmax(output.detach()).item()
        truth = np.argmax(label).item()
        if pred != truth:
            errors[truth] += 1
            wrong_guesses[pred] += 1
        total[truth] += 1
        guesses[pred] += 1
    for i in range(15):
        wrong_guesses[i] /= guesses[i]
        errors[i] /= total[i]
    return errors, wrong_guesses, guesses

if __name__ == "__main__":
    # net = Baseline()
    # train_err, train_loss, val_err, val_loss = train(net, 64, 0.001, 20)
    # plot(train_err, train_loss, val_err, val_loss)
    net = CNN()
    net.load_state_dict(torch.load("./models/bs256_lr0.0001_epoch12", map_location=torch.device('cpu')))
    error_rate, wrong_guess_rate, guesses = performance_per_class(net)
    print(error_rate)
    print(wrong_guess_rate)
    print(guesses)
    plt.plot(error_rate.values())
    plt.title("Error rates per class")
    plt.xlabel("Class")
    plt.ylabel("Error rate")
    plt.show()
    plt.plot(wrong_guess_rate.values())
    plt.title("Wrong guess rate per class")
    plt.xlabel("Class")
    plt.ylabel("Wrong guess rate")
    plt.show()
    plt.plot(guesses.values())
    plt.title("Guesses per class")
    plt.xlabel("Class")
    plt.ylabel("Number of guesses")
    plt.show()
