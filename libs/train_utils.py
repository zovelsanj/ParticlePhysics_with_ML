import torch
import torch.nn as nn
import numpy as np

def train_model(train_dataloader, val_dataloader, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=30):
    train_acc = []
    train_loss = []
    v_acc = []
    v_loss = []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, n_epochs):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_dataloader, model.train(), optimizer, criterion)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))
        train_loss.append(loss)
        train_acc.append(acc)

        # Run **validation**
        val_loss, val_acc = run_epoch(val_dataloader, model.eval(), optimizer, criterion)
        print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
        v_loss.append(val_loss)
        v_acc.append(val_acc)

        # Save model
        # torch.save(model, 't-tagging.pt')
    return val_acc, train_acc, train_loss, v_loss, v_acc

def run_epoch(dataset, model, optimizer, criterion):
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training
    def compute_accuracy(predictions, y):
        return np.mean(np.equal(predictions.detach().numpy(), y.numpy()))
    
    # Iterate through batches
    for data, label in dataset:
        # Grab x and y
        x, y = data[:32], label

        # Get output predictions
        out = model(x)
        
        # Predict and store accuracy
        predictions = torch.argmax(out, dim=1)
        batch_accuracies.append(compute_accuracy(predictions, y))

        # Compute loss
        loss = criterion(out.squeeze(), y.float())
        losses.append(loss.data.item())
        # print(f'loss: {loss}')

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy