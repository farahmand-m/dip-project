# The implementation below is based on the following repository:
# https://github.com/RobotEdh/Yolov-4

import functools
import itertools
import os.path

import numpy as np
import tensorflow as tf
import scipy.stats as stats
import tqdm.auto as tqdm

from modules.common import load_dataset
from modules.common import utils


def tuples_to_tensors(tuples, patch_size, scale_factors):
    for filepath, offset, boxes in tuples:
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        im_height, im_width, channels = image.shape
        target_width = int(np.ceil(im_width / patch_size) * patch_size)
        target_height = int(np.ceil(im_height / patch_size) * patch_size)
        image = tf.image.resize_with_crop_or_pad(image, target_height, target_width)
        maps = [tf.zeros()]
        yield image


def to_tuples(data_dir, images, annotations, patch_size, anchors, scale_factors):
    *annotations, = group_annotations(annotations)
    for image, image_annotations in zip(images, annotations):
        filepath = os.path.join(data_dir, image['file_name'])
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        im_height, im_width, channels = image.shape

        for patch, bounds in utils.to_patches(image, patch_size):
            condition = functools.partial(utils.is_in_bounds, bounds)
            patch_annotations = filter(condition, image_annotations)
            yield patch, to_maps(patch_annotations, anchors, scale_factors)


def create_generator_dataset(data_dir, images, annotations, patch_size, anchors, scale_factors):
    *tuples, = to_tuples(data_dir, images, annotations, patch_size, anchors, scale_factors)
    constructor = functools.partial(tuples_to_tensors, tuples, patch_size)
    signature = tf.TensorSpec(shape=(*patch_size, 3), dtype=tf.float32)
    return tf.data.Dataset.from_generator(constructor, output_signature=signature)


def group_annotations(annotations):
    annotations = sorted(annotations, key=lambda di: di['image_id'])
    current_id = 1
    starting_index = 0
    for current_index, annotation in enumerate(annotations):
        if annotation['image_id'] != current_id:
            yield annotations[starting_index: current_index]
            current_id = current_id + 1
            starting_index = current_index
    yield annotations[starting_index:]


def process(subset, args):
    images = subset['images']
    num_samples = len(images)
    annotations = subset['annotations']
    assert len(annotations) == num_samples
    scale_factors = [8, 16, 32]  # Based on the architecture
    anchors = [(size, int(size * ratio)) for size in args.anchor_sizes for ratio in args.anchor_ratios]
    samples = create_generator_dataset(args.data_dir, images, annotations, args.patch_size, anchors, scale_factors)
    return num_samples, samples


def load_data(args):
    train_set, valid_set, test_set = load_dataset(args.data_dir)
    eval_set = test_set if args.use_test else valid_set
    num_train_samples, train_set = process(train_set, args)
    num_eval_samples, eval_set = process(eval_set, args)
    train_set, eval_set = train_set.repeat(), eval_set.repeat()
    train_set = train_set.shuffle(args.batch_size * 4, seed=args.seed, reshuffle_each_iteration=True)
    train_set, eval_set = train_set.batch(args.batch_size), eval_set.batch(args.batch_size)
    train_steps = np.ceil(num_train_samples / args.batch_size)
    eval_steps = np.ceil(num_eval_samples / args.batch_size)
    train_set, eval_set = train_set.prefetch(4), eval_set.prefetch(4)
    return (train_steps, train_set), (eval_steps, eval_set)
