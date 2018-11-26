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
dataset = Dataset("./out", chardict=chardict, image_size=INPUT_SIZE, feature_map_scale=4, transform=transforms.ToTensor())
loader = data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)

training_model = TrainingModel(backbone=ResNet50Backbone(), recognition=Recognition(vocab=10), detection=Detection())

for images, boxes, ground_truth, texts, target_lengths in loader:
    _, recognition_loss = training_model(images, boxes, texts, target_lengths)
    print(recognition_loss)
    recognition_loss.backward()
    print(texts)
    print(target_lengths)
    break
