import argparse
import os.path
import pathlib
import sys

import tensorflow as tf

from modules import siamese

if __name__ == '__main__':

    default_data_dir = pathlib.Path('data/processed')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=default_data_dir, type=pathlib.Path, help='processed data directory')
    parser.add_argument('--train-steps', type=int, default=None, help='maximum number of training steps')
    parser.add_argument('--eval-steps', type=int, default=None, help='maximum number of evaluation steps')
    parser.add_argument('--use-test', action='store_true', help='weather to use the "test" subset for eval')
    parser.add_argument('--eval-only', action='store_true', help='only perform evaluation using stored weights')
    trainable_modules = {'siamese': siamese}

    subparsers = parser.add_subparsers(dest='module')
    for label, module in trainable_modules.items():
        module.bind_subparser(subparsers)

    args = parser.parse_args()

    module = trainable_modules[args.module]
    checkpoint_path = f'checkpoints/{args.module}_weights.h5'

    print('Loading the dataset... This might take a while.')

    (train_steps, train_set), (eval_steps, eval_set) = module.load_data(args)

    steps_per_epoch = args.train_steps or train_steps
    validation_steps = args.eval_steps or eval_steps

    print('Dataset has been loaded.')

    model = module.create_model(args)
    randomly_initialized = True

    if not args.eval_only:
        print('Beginning Training...')

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor='val_loss', verbose=1,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        mode='min')
        model.fit(train_set,
                  epochs=args.epochs,
                  validation_data=eval_set,
                  steps_per_epoch=steps_per_epoch,
                  validation_steps=validation_steps,
                  callbacks=[checkpoint])
        randomly_initialized = False

    # if os.path.exists(checkpoint_path):
    #     model.load_weights(checkpoint_path)
    #     randomly_initialized = False

    if randomly_initialized:
        print('You must either train the model or have a checkpoint.', file=sys.stderr)
        exit(1)

    print('Beginning Evaluation...')

    module.evaluate_model(model, eval_set, validation_steps)
