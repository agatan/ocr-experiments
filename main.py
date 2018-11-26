import os
import json
import math
from typing import List

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import datagen
from data import CharDictionary, Dataset
from model import TrainingModel, ResNet50Backbone, Recognition, Detection


INPUT_SIZE = np.array([192, 288])
FEATURE_SIZE = INPUT_SIZE // 4


chardict = CharDictionary("0123456789")
dataset = Dataset("./data/train", chardict=chardict, image_size=INPUT_SIZE, feature_map_scale=4, transform=transforms.ToTensor())
loader = data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)

training_model = TrainingModel(backbone=ResNet50Backbone(), recognition=Recognition(vocab=10), detection=Detection())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

training_model.to(device)
optimizer = torch.optim.Adam(training_model.parameters())

nan_found = False
for epoch in range(10):
    training_model.train()
    for images, boxes, ground_truth, texts, target_lengths in loader:
        training_model.zero_grad()
        images = images.to(device)
        boxes = boxes.to(device)
        ground_truth = ground_truth.to(device)
        texts = texts.to(device)
        target_lengths = target_lengths.to(device)
        _, recognition_loss = training_model(images, boxes, texts, target_lengths)
        print(recognition_loss.detach().item())
        recognition_loss.backward()
        for name, p in training_model.named_parameters():
            if p.grad is not None and torch.any(p.grad != p.grad):
                nan_count = torch.sum(p.grad != p.grad).item()
                count = torch.sum(torch.ones_like(p.grad, dtype=torch.long)).item()
                print(name, nan_count / float(count))
                nan_found = True
        optimizer.step()
        if nan_found:
            break
    if nan_found:
        break
