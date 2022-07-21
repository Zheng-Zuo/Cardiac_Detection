import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import cv2
import imgaug.augmenters as iaa
from dataset import CardiacDataset
from train import CardiacDetectionModel


model = CardiacDetectionModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.load_from_checkpoint("weight.ckpt")
model.eval()
model.to(device)

val_root_path = "Processed-Heart-Detection/val/"
val_subjects = "val_subjects.npy"

val_dataset = CardiacDataset("rsna_heart_detection.csv", val_subjects, val_root_path, None)

preds = []
labels = []

with torch.no_grad():
    for data, label in val_dataset:
        data = data.to(device).float().unsqueeze(0)
        pred = model(data)[0].cpu()
        preds.append(pred)
        labels.append(label)
        
preds=torch.stack(preds)
labels=torch.stack(labels)

print(abs(preds-labels).mean(0))