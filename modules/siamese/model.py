# The implementation below is based on the following tutorial:
# https://keras.io/examples/vision/siamese_network

import numpy as np
import scipy.spatial.distance as distance
import tensorflow as tf


class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor embedding and the positive embedding, and
    the anchor embedding and the negative embedding.
    """

    # noinspection PyMethodOverriding
    def call(self, anchor, positive, negative):
        anchor_positive_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        anchor_negative_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return anchor_positive_distance, anchor_negative_distance


class SiameseModel(tf.keras.models.Model):
    """
    The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, im_size, margin):
        super(SiameseModel, self).__init__()
        self.margin = margin

        image_shape = (*im_size, 3)

        # Our Siamese Network will generate embeddings for each of the images of the triplet. To do this, we will use a
        # ResNet50 model pretrained on ImageNet and connect a few Dense layers to it, so we can learn to separate these
        # embeddings.

        base = tf.keras.applications.resnet.ResNet50(weights='imagenet', input_shape=image_shape, include_top=False)

        x = tf.keras.layers.Flatten()(base.output)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256)(x)

        self.embedding = tf.keras.Model(base.input, x, name='Embedding')

        # We will freeze the weights of all the layers of the model up until the layer conv5_block1_out. This is important
        # to avoid affecting the weights that the model has already learned. We are going to leave the bottom few layers
        # trainable, so that we can fine-tune their weights during training.

        trainable = False
        for layer in base.layers:
            if layer.name == 'conv5_block1_out':
                trainable = True
            layer.trainable = trainable

        # The Siamese network will receive each of the triplet images as an input, generate the embeddings, and output
        # the distance between the anchor and the positive embedding, as well as the distance between the anchor and
        # the negative embedding.

        anchor = tf.keras.layers.Input(name='anchor', shape=image_shape)
        positive = tf.keras.layers.Input(name='positive', shape=image_shape)
        negative = tf.keras.layers.Input(name='negative', shape=image_shape)

        inputs = [anchor, positive, negative]

        anchor_embedding = self.embedding(tf.keras.applications.resnet.preprocess_input(anchor))
        positive_embedding = self.embedding(tf.keras.applications.resnet.preprocess_input(positive))
        negative_embedding = self.embedding(tf.keras.applications.resnet.preprocess_input(negative))

        distances = DistanceLayer()(anchor_embedding, positive_embedding, negative_embedding)

        self.distances = tf.keras.Model(inputs=inputs, outputs=distances)

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def build(self, im_size):
        triplet_shape = [(None, *im_size, 3)] * 3
        super().build(input_shape=triplet_shape)

    def call(self, inputs, training=False, **kwargs):
        # To simplify the implementation of the loss function, we can use the `distances` network here.
        return self.distances(inputs)

    def loss_func(self, data):
        # The output of the network is a tuple containing the distances between the anchor and the positive example,
        # and the anchor and the negative example.
        anchor_positive_distance, anchor_negative_distance = self.distances(data)

        # Computing the Triplet Loss by subtracting both distances and making sure we don't get a negative value.
        loss = anchor_positive_distance - anchor_negative_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that you do inside. We are using it here to
        # compute the loss, so we can get the gradients and apply them using the optimizer specified in `compile()`.
        with tf.GradientTape() as tape:
            loss = self.loss_func(data)

        # Storing the gradients of the loss function with respect to the weights/parameters.
        gradients = tape.gradient(loss, self.distances.trainable_weights)

        # Applying the gradients on the model using the specified optimizer; a.k.a., backprop!
        self.optimizer.apply_gradients(zip(gradients, self.distances.trainable_weights))

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.loss_func(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def predict_step(self, data):
        anchors, positives, negatives = data
        anchors = tf.keras.applications.resnet.preprocess_input(anchors)
        positives = tf.keras.applications.resnet.preprocess_input(positives)
        negatives = tf.keras.applications.resnet.preprocess_input(negatives)
        return self.embedding(anchors), self.embedding(positives), self.embedding(negatives)

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be called automatically.
        return [self.loss_tracker]


def create_model(args):
    # Note that we  need to implement a model with custom training loop, so we can compute the triplet loss using the
    # three embeddings produced by the Siamese network. We can achieve this by subclassing `tf.keras.models.Model` and
    # overriding a few specific methods.

    model = SiameseModel(args.im_size, margin=1.0)

    # The model can now be trained. We will use the Adam optimizer with a relatively low learning rate.

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    # We have to set `weighted_metrics` to silent a pesky warning!
    model.compile(optimizer, weighted_metrics=[])
    return model


def evaluate_model(model, subset, steps):
    anchors_embed, positives_embed, negatives_embed = model.predict(subset, steps=steps, verbose=1)

    print('Accuracy:')

    positives_dist = np.array([distance.euclidean(u, v) for u, v in zip(anchors_embed, positives_embed)])
    negatives_dist = np.array([distance.euclidean(u, v) for u, v in zip(anchors_embed, negatives_embed)])

    print('  Euclidean Distance:', np.mean(positives_dist < negatives_dist))

    positives_dist = np.array([distance.cosine(u, v) for u, v in zip(anchors_embed, positives_embed)])
    negatives_dist = np.array([distance.cosine(u, v) for u, v in zip(anchors_embed, negatives_embed)])

    print('  Cosine Similarity: ', np.mean(positives_dist < negatives_dist))
