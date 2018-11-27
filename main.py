import os
import json
import math
import random
from typing import List
import logging
import shutil

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.utils
import torchvision.transforms as transforms

import datagen
from data import CharDictionary, Dataset
from model import TrainingModel, Recognition, Detection
from backbone import ResNet50Backbone


INPUT_SIZE = np.array([192, 288])
FEATURE_SIZE = INPUT_SIZE // 4
logger = logging.getLogger(__name__)


def save_checkpoint(state, filename, is_best):
    logger.info("Saving the state to {}".format(filename))
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(os.path.dirname(filename), "best.pth")
        if filename.endswith(".tar"):
            best_file += ".tar"
        logger.info("Copy {} to {}".format(filename, best_file))
        shutil.copyfile(filename, best_file)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=str)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--charset", choices=["digits"], default="digits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", default=False, action='store_true')
    parser.add_argument("--checkpoint", default="checkpoint")
    parser.add_argument("--confidence_loss_function", default="bce", choices=["bce", "focalloss"])
    parser.add_argument("--restore")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
    )

    logger.info("random seed is {}".format(args.seed))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("charset is {}".format(args.charset))
    if args.charset == "digits":
        chardict = CharDictionary("0123456789")
    else:
        raise NotImplementedError("charset {} is not implemented yet".format(args.charset))

    logger.info("Loading training dataset from {}".format(args.train))
    dataset = Dataset(args.train, chardict=chardict, image_size=INPUT_SIZE, feature_map_scale=4, transform=transforms.ToTensor())
    loader = data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("device: {}".format(device))

    logger.info("Instantiate training model")
    backbone = ResNet50Backbone()
    recognition = Recognition(vocab=chardict.vocab)
    detection = Detection()
    training_model = TrainingModel(backbone=backbone, recognition=recognition, detection=detection, confidence_loss_function=args.confidence_loss_function)
    training_model.to(device)
    optimizer = torch.optim.Adam(training_model.parameters())

    start_epoch = 0
    step = 0
    if args.restore:
        logger.info("Restore from {}".format(args.restore))
        state = torch.load(args.restore)
        start_epoch = state['epoch']
        step = state['step']
        backbone.load_state_dict(state['backbone'])
        recognition.load_state_dict(state['recognition'])
        detection.load_state_dict(state['detection'])
        optimizer.load_state_dict(state['optimizer'])
        logger.info("Loaded checkpoint {} (resume from epoch {}, step {})".format(args.restore, start_epoch, step))

    os.makedirs(args.checkpoint, exist_ok=True)
    nan_found = False
    for epoch in range(start_epoch, args.epochs):
        logger.info("[Epoch {}]".format(epoch))
        training_model.train()
        for images, boxes, ground_truth, texts, target_lengths in loader:
            step += 1
            training_model.zero_grad()
            images = images.to(device)
            boxes = boxes.to(device)
            ground_truth = ground_truth.to(device)
            texts = texts.to(device)
            target_lengths = target_lengths.to(device)
            confidence_loss, regression_loss, confidences_accuracy, recognition_loss = training_model(images, boxes, ground_truth, texts, target_lengths)
            logger.info("Confidence Loss: {}, Regression Loss: {}, Recognition Loss: {}".format(
                confidence_loss.detach().item(),
                regression_loss.detach().item(),
                recognition_loss.detach().item(),
            ))
            logger.info("Confidence Accracuy: {:.4f}%".format(confidences_accuracy.item()))
            loss = confidence_loss + regression_loss + recognition_loss
            loss.backward()
            for name, p in training_model.named_parameters():
                if p.grad is not None and torch.any(p.grad != p.grad):
                    nan_count = torch.sum(p.grad != p.grad).item()
                    count = torch.sum(torch.ones_like(p.grad, dtype=torch.long)).item()
                    print(name, nan_count / float(count))
                    nan_found = True
            optimizer.step()
            if nan_found:
                torchvision.utils.save_image(images, "out.png")
                break
        save_checkpoint({
            'step': step,
            'epoch': epoch,
            'backbone': backbone.state_dict(),
            'detection': detection.state_dict(),
            'recognition': recognition.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.checkpoint, "epoch-{}-step-{}.pth.tar".format(epoch, step)), is_best=False)
        if nan_found:
            break


if __name__ == "__main__":
    main()
