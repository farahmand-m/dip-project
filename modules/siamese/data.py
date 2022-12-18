# The implementation below is based on the following tutorial:
# https://keras.io/examples/vision/siamese_network

import functools

import numpy as np
import tensorflow as tf
import scipy.stats as stats
import tqdm.auto as tqdm

from modules.common import load_dataset


def group_tracks(annotations):
    current_track_id = 1
    starting_index = 0
    for current_index, annotation in enumerate(annotations):
        if annotation['track_id'] != current_track_id:
            yield annotations[starting_index: current_index]
            current_track_id = annotation['track_id']
            starting_index = current_index
    yield annotations[starting_index:]


def get_pair(track_index, current_track, current_box, tracks, images, rng, data_dir, same_track, same_box):
    other_tracks = np.array(tracks[:track_index] + tracks[track_index + 1:], dtype=object)
    target_track = current_track if same_track else rng.choice(other_tracks)
    target_box = current_box if same_box else rng.choice(target_track)
    image_id = target_box['image_id']
    image = images[image_id - 1]
    filepath = data_dir / image['file_name']
    return str(filepath), target_box['bbox']


def pairs_to_crops(pairs, target_size):
    for filepath, bounding_box in pairs:
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        im_height, im_width, channels = image.shape
        left, top, width, height = bounding_box
        left = np.clip(left, 0, im_width)
        top = np.clip(top, 0, im_height)
        width -= left + width - np.clip(left + width, 0, im_width)
        height -= top + height - np.clip(top + height, 0, im_height)
        image = tf.image.crop_to_bounding_box(image, top, left, height, width)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_size)
        yield image


def create_generator_dataset(tracks, images, rng, data_dir, target_size, same_track, same_box):
    candidates = [(track_index, track, box) for track_index, track in enumerate(tracks) for box in track]
    pairs = [get_pair(*tup, tracks, images, rng, data_dir, same_track, same_box) for tup in tqdm.tqdm(candidates)]
    constructor = functools.partial(pairs_to_crops, pairs, target_size)
    signature = tf.TensorSpec(shape=(*target_size, 3), dtype=tf.float32)
    return len(pairs), tf.data.Dataset.from_generator(constructor, output_signature=signature)


def process(subset, args):
    rng = np.random.default_rng(seed=args.seed)
    images = subset['images']
    annotations = subset['annotations']
    *tracks, = group_tracks(annotations)
    num_samples, anchors = create_generator_dataset(tracks, images, rng, args.data_dir, args.im_size, same_track=True, same_box=True)
    num_samples, positives = create_generator_dataset(tracks, images, rng, args.data_dir, args.im_size, same_track=True, same_box=False)
    num_samples, negatives = create_generator_dataset(tracks, images, rng, args.data_dir, args.im_size, same_track=False, same_box=False)
    return num_samples, (anchors, positives, negatives)


def load_data(args):
    train_set, valid_set, test_set = load_dataset(args.data_dir)
    eval_set = test_set if args.use_test else valid_set
    num_train_samples, train_set = process(train_set, args)
    num_eval_samples, eval_set = process(eval_set, args)
    train_set, eval_set = tf.data.Dataset.zip(train_set), tf.data.Dataset.zip(eval_set)
    train_set = train_set.shuffle(args.batch_size * 4, seed=args.seed, reshuffle_each_iteration=True)
    train_set, eval_set = train_set.batch(args.batch_size), eval_set.batch(args.batch_size)
    train_steps = np.ceil(num_train_samples / args.batch_size)
    eval_steps = np.ceil(num_eval_samples / args.batch_size)
    train_set, eval_set = train_set.prefetch(1), eval_set.prefetch(1)
    return (train_steps, train_set), (eval_steps, eval_set)
