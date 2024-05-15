from utils import *
from feature_extraction import *

from torch.utils.data import DataLoader
from torcheeg.models import DGCNN

from torcheeg.trainers import ClassifierTrainer
from torcheeg.model_selection import KFold

import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


if __name__ == '__main__':
    data_path = 'E:\\Extracted_feature_base'
    all_data, all_labels = get_source_data_detect(data_path)
    
    action_labels = [0, 1, 2, 3]
    baseline_label = 4

    for action_label in action_labels:
        # Filter data for current action and baseline
        train_indices = np.where((all_labels == action_label) | (all_labels == baseline_label))[0]
        train_data = all_data[train_indices]
        train_labels = all_labels[train_indices]
        
        # Convert labels to binary
        train_labels = np.where(train_labels == action_label, 1, 0)

        # Split dataset into training and testing sets
        split_index = int(0.7 * len(train_data))
        train_data, test_data = train_data[:split_index], train_data[split_index:]
        train_labels, test_labels = train_labels[:split_index], train_labels[split_index:]

        # Convert data to tensors
        train_data = torch.tensor(train_data, dtype=torch.float32).squeeze(1)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.float32).squeeze(1)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = DGCNN(in_channels=23, num_electrodes=4, hid_channels=64, num_classes=2)
        
        trainer = ClassifierTrainer(model=model,
                                    num_classes=2,
                                    lr=1e-4,
                                    weight_decay=1e-4,
                                    accelerator="gpu")

        trainer.fit(train_loader,
                    test_loader,
                    max_epochs=60,
                    default_root_dir=f'./examples_amigos_dgcnn/model/action_{action_label}_vs_baseline',
                    callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                    enable_progress_bar=True,
                    enable_model_summary=True,
                    limit_val_batches=0.0)

        score = trainer.test(test_loader,
                             enable_progress_bar=True,
                             enable_model_summary=True)[0]

        print(f'test accuracy for action {action_label} vs baseline: {score["test_accuracy"]:.4f}')