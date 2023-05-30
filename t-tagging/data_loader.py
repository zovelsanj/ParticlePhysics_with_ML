import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class TabularDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        labels = list(self.dataset['is_signal_new'])
        label = labels[idx] 
        sample_data = list(self.dataset.iloc[idx, :-1])
        data = torch.tensor(sample_data, dtype=torch.float32)
        return data, label

def process_dataset(data_path):
    dataset = load_dataset(path)
    #we won’t need the top-quark 4-vector columns and ttv so,
    dataset = dataset.remove_columns(['truthE', 'truthPX', 'truthPY', 'truthPZ', 'ttv'])
    dataset.set_format("pandas")
    train_df, test_df = dataset["train"][:], dataset["test"][:]
    return train_df, test_df

def dataset_split(train_dataset, test_dataset, batch_size):
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader
