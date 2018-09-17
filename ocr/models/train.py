"""Run training.

This is a script to run training and save the trained models.
"""
from argparse import ArgumentParser

import torch
from ocr.data import process
from ocr.models import net
from ocr.preprocessing.dataset import CSVDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/train/annotations.csv")
    parser.add_argument(
        "--validation_csv", default="data/processed/validation/annotations.csv"
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--checkpoint_path", default="checkpoint-weights.h5", type=str)
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out", "-o", default="weights.h5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = CSVDataset(
        args.train_csv, features_pixel=4, input_size=(192, 256), aug=True
    )
    loader = torch.utils.data.DataLoader(gen, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_gen = CSVDataset(
        args.validation_csv, features_pixel=4, input_size=(192, 256),
    )
    valid_loader = torch.utils.data.DataLoader(valid_gen, batch_size=4, shuffle=False)

    ocr_net = net.OCRNet().to(device)
    optimizer = torch.optim.Adam(ocr_net.parameters(), lr=1e-2)
    if args.weights:
        ocr_net.load_state_dict(torch.load(args.weights))

    def step(engine, batch):
        images, bbox_true = batch
        images = images.to(device)
        bbox_true = bbox_true.to(device)
        ocr_net.zero_grad()
        bbox_pred = ocr_net(images)
        loss_confidence = ocr_net.loss_confidence(bbox_pred, bbox_true)
        ious = ocr_net.ious(bbox_pred, bbox_true)
        loss_iou = -torch.mean(torch.log(ious + 1e-5))
        loss = loss_confidence + loss_iou
        loss.backward()
        optimizer.step()
        return {
            'loss_confidence': loss_confidence.item(),
            'loss_iou': loss_iou.item(),
            'iou': torch.mean(ious).item(),
        }

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(args.checkpoint_path, "networks", save_interval=1, n_saved=5, require_empty=False)
    timer = Timer(average=True)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={
        'ocr_net': ocr_net,
    })
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    running_avgs = {}
    @trainer.on(Events.ITERATION_COMPLETED)
    def update_logs(engine):
        alpha = 0.90
        for k, v in engine.state.output.items():
            old_v = running_avgs.get(k, v)
            new_v = alpha * old_v + (1 - alpha) * v

            running_avgs[k] = new_v

    PRINT_FREQ = 100
    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):
        if (engine.state.iteration - 1) % PRINT_FREQ == 0:
            columns = running_avgs.keys()
            values = [str(round(value, 5)) for value in running_avgs.values()]

            message = '[{epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                      i=(engine.state.iteration % len(loader)),
                                                      max_i=len(loader))
            for name, value in zip(columns, values):
                message += ' | {name}: {value}'.format(name=name, value=value)
            print(message)
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        print('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))
        timer.reset()

    trainer.run(loader, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
