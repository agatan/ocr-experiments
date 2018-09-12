"""Run training.

This is a script to run training and save the trained models.
"""
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python import debug as tf_debug, debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

from ocr.data import process
from ocr.models import resnet50, mobilenet, bboxnet_subclass
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
    # callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5))
    # callbacks.append(tf.keras.callbacks.EarlyStopping(patience=30))
    return callbacks


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/train/annotations.csv")
    parser.add_argument(
        "--validation_csv", default="data/processed/validation/annotations.csv"
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
    args = parser.parse_args()

    if args.debug:
        set_debugger_session()

    features_pixel = 8

    gen = CSVGenerator(
        args.train_csv, features_pixel=features_pixel, input_size=(512 // 2, 832 // 2), aug=True
    )
    valid_gen = CSVGenerator(
        args.validation_csv, features_pixel=features_pixel, input_size=(512 // 2, 832 // 2), aug=True
    )
    config = tf.estimator.RunConfig(model_dir=args.checkpoint_dir, save_checkpoints_steps=100, keep_checkpoint_max=3)
    estimator = tf.estimator.Estimator(model_fn=bboxnet_subclass.model_fn, model_dir=args.checkpoint_dir, config=config)
    if args.debug:
        hooks = [debug.LocalCLIDebugHook()]
    else:
        hooks = [tf.train.StepCounterHook()]
    train_input_fn = bboxnet_subclass.make_input_fn(gen, batch_size=args.batch_size)
    eval_input_fn = bboxnet_subclass.make_input_fn(valid_gen, batch_size=args.batch_size)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=args.max_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    main()
