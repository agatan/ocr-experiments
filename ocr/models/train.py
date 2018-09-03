"""Run training.

This is a script to run training and save the trained models.
"""
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

from ocr.data import process
from ocr.models import resnet50, mobilenet
from ocr.models.bboxnet import create_model
from ocr.preprocessing.generator import CSVGenerator

K = tf.keras.backend


def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", has_inf_or_nan)
    K.set_session(sess)


def create_callbacks(checkpoint_path: str, log_dir: str):
    callbacks = []
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, save_weights_only=True
        )
    )
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir))
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5))
    return callbacks


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
    parser.add_argument("--out", "-o", default="weights.h5")
    parser.add_argument("--weight", default=None, type=str)
    parser.add_argument("--backbone", default="mobilenet", type=str)
    args = parser.parse_args()

    if args.debug:
        set_debugger_session()

    if args.backbone == "resnet50":
        backbone, features_pixel = resnet50.backbone((512, 832, 3))
    elif args.backbone == "mobilenet":
        backbone, features_pixel = mobilenet.backbone((512, 832, 3))
    else:
        raise ValueError("Unknown backobne {}".format(args.backbone))
    training_model, _ = create_model(
        backbone,
        features_pixel=features_pixel,
        input_shape=(512, 832, 3),
        n_vocab=process.vocab(),
    )
    if args.weight:
        training_model.load_weights(args.weight)

    gen = CSVGenerator(
        args.train_csv, features_pixel=features_pixel, input_size=(512, 832), aug=True
    )
    valid_gen = CSVGenerator(
        args.validation_csv, features_pixel=features_pixel, input_size=(512, 832)
    )
    steps_per_epoch = (gen.size() - 1) // args.batch_size + 1

    callbacks = create_callbacks(args.checkpoint_path, args.logdir)
    training_model.fit_generator(
        gen.batches(args.batch_size, infinite=True),
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen.batches(4),
        validation_steps=50,
        callbacks=callbacks,
    )
    training_model.save_weights(args.out)


if __name__ == "__main__":
    main()
