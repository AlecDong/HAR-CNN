from preprocessing import data_loader
from model import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

net = CNN()
def train(net, batch_size=32, lr=0.001, num_epochs=30):
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(0)

    train_loader, val_loader = data_loader(batch_size=batch_size)
    
    # cross entropy loss function and adaptive moment estimation optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

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
        for batch in train_loader:
            train_batches += 1
            imgs, labels = batch.values()
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = softmax(outputs)

            # find error and loss for training data
            total_train_err += (np.argmax(pred.detach(), 1) != labels).sum().item()
            total_train_loss += loss.item()
            train_iters += len(labels)

        val_batches = 0
        for batch in val_loader:
            val_batches += 1
            imgs, labels = batch.values()
            outputs = net(imgs)
            loss = criterion(outputs, labels)

            pred = softmax(outputs)

            # find error and loss for training data
            total_val_err += (np.argmax(pred.detach(), 1) != labels).sum().item()
            total_val_loss += loss.item()
            val_iters += len(labels)

        # record the average error (per iteration) and loss (per batch) for each epoch
        train_err[epoch] = total_train_err / train_iters
        train_loss[epoch] = total_train_loss / train_batches
        val_err[epoch] = total_val_err / val_iters
        val_loss[epoch] = total_val_loss / val_batches
        print(f"Epoch {epoch}: Train err: {train_err[epoch]}% Val err: {val_err[epoch]}%")

train(net)