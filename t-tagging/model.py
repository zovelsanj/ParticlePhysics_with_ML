import torch.nn as nn
from libs import train_utils
from data_loader import process_dataset, TabularDataset, dataset_split

class LinearBlock(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=out_features)
        )

class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()
        
        self.t_tagger = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            LinearBlock(in_features=in_channels, out_features=200),
            LinearBlock(in_features=200, out_features=50),
            nn.Linear(in_features=50, out_features=out_channels)
        )
        
    def forward(self, x):
        y = self.t_tagger(x)
        return y

if __name__ == "__main__":
    train_df, test_df = process_dataset("dl4phys/top_tagging")
    train_dataset = TabularDataset(train_df)
    test_dataset = TabularDataset(test_df)
    model = Model(in_channels=800, out_channels=1)

    train_dataloader, val_dataloader, test_dataloader = dataset_split(train_dataset, test_dataset, batch_size=32)
    # Training the model
    train_model(train_dataloader, val_dataloader, model, lr=0.01, momentum=0.9, n_epochs=30)

    # Testing the model
    loss, accuracy = run_epoch(test_dataloader, model.eval(), None)
    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))