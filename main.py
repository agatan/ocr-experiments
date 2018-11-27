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
from tensorboardX import SummaryWriter

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--charset", choices=["digits"], default="digits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", default=False, action='store_true')
    parser.add_argument("--checkpoint", default="checkpoint")
    parser.add_argument("--confidence_loss_function", default="bce", choices=["bce", "focalloss"])
    parser.add_argument("--restore")
    parser.add_argument("--logdir", default="logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
    )

    logger.info("logdir is {}".format(args.logdir))
    writer = SummaryWriter(log_dir=args.logdir)

    logger.info("random seed is {}".format(args.seed))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)

    logger.info("charset is {}".format(args.charset))
    if args.charset == "digits":
        chardict = CharDictionary("0123456789")
    else:
        raise NotImplementedError("charset {} is not implemented yet".format(args.charset))

    logger.info("Loading training dataset from {}".format(args.train))
    dataset = Dataset(args.train, chardict=chardict, image_size=INPUT_SIZE, feature_map_scale=4, transform=transforms.ToTensor())
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("device: {}".format(device))

    logger.info("Instantiate training model")
    backbone = ResNet50Backbone(pretrained=True)
    recognition = Recognition(vocab=chardict.vocab)
    detection = Detection()
    training_model = TrainingModel(backbone=backbone, recognition=recognition, detection=detection, confidence_loss_function=args.confidence_loss_function)
    training_model.to(device)
    optimizer = torch.optim.Adam(training_model.parameters())

    def lr_sched(epoch):
        if epoch < 2:
            return 1e-3
        elif epoch < 5:
            return 5e-4
        return 1e-4
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    start_epoch = 0
    step = 0
    best_loss = None
    if args.restore:
        logger.info("Restore from {}".format(args.restore))
        state = torch.load(args.restore)
        start_epoch = state['epoch'] + 1
        step = state['step']
        best_loss = state['best_loss']
        backbone.load_state_dict(state['backbone'])
        recognition.load_state_dict(state['recognition'])
        detection.load_state_dict(state['detection'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        logger.info("Loaded checkpoint {} (resume from epoch {}, step {})".format(args.restore, start_epoch, step))

    os.makedirs(args.checkpoint, exist_ok=True)
    confidence_losses = []
    confidence_accuracies = []
    regression_losses = []
    recognition_losses = []
    losses = []

    for epoch in range(start_epoch, args.epochs):
        logger.info("[Epoch {}]".format(epoch))
        scheduler.step()
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
            confidence_losses.append(confidence_loss.detach().item())
            confidence_accuracies.append(confidences_accuracy.detach().item())
            regression_losses.append(regression_loss.detach().item())
            recognition_losses.append(recognition_loss.detach().item())
            loss = confidence_loss + regression_loss + recognition_loss
            losses.append(loss.detach().item())
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                writer.add_scalar("learning_rate", lr_sched(epoch), step)
                writer.add_scalar("loss", np.mean(losses), step)
                writer.add_scalar("loss/confidence", np.mean(confidence_losses), step)
                writer.add_scalar("loss/regression", np.mean(regression_losses), step)
                writer.add_scalar("loss/recognition", np.mean(recognition_losses), step)
                writer.add_scalar("accuracy/confidence", np.mean(confidence_accuracies), step)
                logger.info("[Epoch {} Step {} / {}]".format(epoch, step, len(dataset) // args.batch_size))
                logger.info("Confidence Loss: {}, Regression Loss: {}, Recognition Loss: {}".format(
                    np.mean(confidence_losses),
                    np.mean(regression_losses),
                    np.mean(recognition_losses),
                ))
                logger.info("Confidence Accracuy: {:.4f}%".format(np.mean(confidence_accuracies)))
                confidence_losses = []
                confidence_accuracies = []
                regression_losses = []
                recognition_losses = []
                mean_loss = np.mean(losses)
                losses = []

                if best_loss is None or best_loss > mean_loss:
                    best_loss = mean_loss
                    save_checkpoint({
                        'step': step,
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'backbone': backbone.state_dict(),
                        'detection': detection.state_dict(),
                        'recognition': recognition.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, os.path.join(args.checkpoint, "best.pth.tar"))
        if nan_found:
            break
    writer.close()


if __name__ == "__main__":
    main()
