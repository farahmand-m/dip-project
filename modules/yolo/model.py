# The implementation below is based on the following repository:
# https://github.com/Tianxiaomo/pytorch-YOLOv4
# https://github.com/RobotEdh/Yolov-4

import functools

import tensorflow as tf
import tensorflow_addons as tfa


class CustomConv2D(tf.keras.models.Model):

    def __init__(self, filters, kernel_size, stride, use_mish=True, use_bias=True, use_batchnorm=True, linear_activation=False):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding='same', use_bias=use_bias)
        self.batchnorm = tf.keras.layers.BatchNormalization() if use_batchnorm else tf.identity
        leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        activation = tfa.activations.mish if use_mish else leaky_relu
        activation = tf.identity if linear_activation else activation
        self.mish = tf.keras.layers.Activation(activation)

    def call(self, x, training=None, mask=None):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.mish(x)


class ResidualBlocks(tf.keras.models.Model):

    def __init__(self, num_blocks, filters, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.blocks = [tf.keras.Sequential([
            CustomConv2D(filters, 1, 1), CustomConv2D(filters, 3, 1)
        ]) for _ in range(num_blocks)]

    def call(self, x, training=None, mask=None):
        for sequential in self.blocks:
            h = sequential(x)
            x = x + h if self.skip_connection else h
        return x


class BackboneBase(tf.keras.models.Model):

    def __init__(self, filters):
        super().__init__()
        filters_halved = filters // 2
        self.conv1 = CustomConv2D(filters_halved, 3, 1)
        self.conv2 = CustomConv2D(filters, 3, 2)
        self.conv3 = CustomConv2D(filters, 1, 1)
        self.conv4 = CustomConv2D(filters, 1, 1)
        self.conv5 = CustomConv2D(filters_halved, 1, 1)
        self.conv6 = CustomConv2D(filters, 3, 1)
        self.conv7 = CustomConv2D(filters, 1, 1)
        self.conv8 = CustomConv2D(filters, 1, 1)

    def call(self, x, training=None, mask=None):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = x6 + x4
        x7 = self.conv7(x6)
        x7 = tf.concat([x7, x3], axis=-1)
        return self.conv8(x7)


class BackboneBlock(tf.keras.models.Model):

    def __init__(self, filters, num_res_blocks):
        super().__init__()
        filters_halved = filters // 2
        self.conv1 = CustomConv2D(filters, 3, 2)
        self.conv2 = CustomConv2D(filters_halved, 1, 1)
        self.conv3 = CustomConv2D(filters_halved, 1, 1)
        self.residual = ResidualBlocks(num_res_blocks, filters_halved)
        self.conv4 = CustomConv2D(filters_halved, 1, 1)
        self.conv5 = CustomConv2D(filters, 1, 1)

    def call(self, x, training=None, mask=None):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        residual_x = self.residual(x3)
        x4 = self.conv4(residual_x)
        x4 = tf.concat([x4, x2], axis=-1)
        return self.conv5(x4)


class Neck(tf.keras.models.Model):

    def __init__(self, filters):
        super().__init__()
        self.conv1 = CustomConv2D(filters // 2, 1, 1, use_mish=False)
        self.conv2 = CustomConv2D(filters, 3, 1, use_mish=False)
        self.conv3 = CustomConv2D(filters // 2, 1, 1, use_mish=False)
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=5, strides=1, padding='same')
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=9, strides=1, padding='same')
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=13, strides=1, padding='same')
        self.conv4 = CustomConv2D(filters // 2, 1, 1, use_mish=False)
        self.conv5 = CustomConv2D(filters, 3, 1, use_mish=False)
        self.conv6 = CustomConv2D(filters // 2, 1, 1, use_mish=False)
        self.conv7 = CustomConv2D(filters // 4, 1, 1, use_mish=False)
        self.upsample1 = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')
        self.conv8 = CustomConv2D(filters // 4, 1, 1, use_mish=False)
        self.conv9 = CustomConv2D(filters // 4, 1, 1, use_mish=False)
        self.conv10 = CustomConv2D(filters // 2, 3, 1, use_mish=False)
        self.conv11 = CustomConv2D(filters // 4, 1, 1, use_mish=False)
        self.conv12 = CustomConv2D(filters // 2, 3, 1, use_mish=False)
        self.conv13 = CustomConv2D(filters // 4, 1, 1, use_mish=False)
        self.conv14 = CustomConv2D(filters // 8, 1, 1, use_mish=False)
        self.upsample2 = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')
        self.conv15 = CustomConv2D(filters // 8, 1, 1, use_mish=False)
        self.conv16 = CustomConv2D(filters // 8, 1, 1, use_mish=False)
        self.conv17 = CustomConv2D(filters // 4, 3, 1, use_mish=False)
        self.conv18 = CustomConv2D(filters // 8, 1, 1, use_mish=False)
        self.conv19 = CustomConv2D(filters // 4, 3, 1, use_mish=False)
        self.conv20 = CustomConv2D(filters // 8, 1, 1, use_mish=False)

    def call(self, inputs, training=None, mask=None):
        i1, i2, i3 = inputs
        x1 = self.conv1(i1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = tf.concat([m3, m2, m1, x3], axis=-1)
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        up = self.upsample1(x7)
        x8 = self.conv8(i2)
        x8 = tf.concat([x8, up], axis=-1)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        up = self.upsample2(x14)
        x15 = self.conv15(i3)
        x15 = tf.concat([x15, up], axis=-1)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Head(tf.keras.models.Model):

    def __init__(self, filters, num_classes, num_anchors):
        super().__init__()

        output_dim = (4 + 1 + num_classes) * num_anchors  # Coordinates and confidence, respectively.
        output_layers_kwargs = {'use_bias': True, 'linear_activation': True, 'use_batchnorm': False}

        self.conv1 = CustomConv2D(filters // 4, 3, 1, use_mish=False)
        self.conv2 = CustomConv2D(output_dim, 1, 1, **output_layers_kwargs)

        self.conv3 = CustomConv2D(filters // 4, 3, 2, use_mish=False)
        self.conv4 = CustomConv2D(filters // 4, 1, 1, use_mish=False)
        self.conv5 = CustomConv2D(filters // 2, 3, 1, use_mish=False)
        self.conv6 = CustomConv2D(filters // 4, 1, 1, use_mish=False)
        self.conv7 = CustomConv2D(filters // 2, 3, 1, use_mish=False)
        self.conv8 = CustomConv2D(filters // 4, 1, 1, use_mish=False)
        self.conv9 = CustomConv2D(filters // 2, 3, 1, use_mish=False)
        self.conv10 = CustomConv2D(output_dim, 1, 1, **output_layers_kwargs)

        self.conv11 = CustomConv2D(filters // 2, 3, 2, use_mish=False)
        self.conv12 = CustomConv2D(filters // 2, 1, 1, use_mish=False)
        self.conv13 = CustomConv2D(filters, 3, 1, use_mish=False)
        self.conv14 = CustomConv2D(filters // 2, 1, 1, use_mish=False)
        self.conv15 = CustomConv2D(filters, 3, 1, use_mish=False)
        self.conv16 = CustomConv2D(filters // 2, 1, 1, use_mish=False)
        self.conv17 = CustomConv2D(filters, 3, 1, use_mish=False)
        self.conv18 = CustomConv2D(output_dim, 1, 1, **output_layers_kwargs)

    def call(self, inputs, training=None, mask=None):
        i1, i2, i3 = inputs
        x1 = self.conv1(i1)
        x2 = self.conv2(x1)
        x3 = self.conv3(i1)
        x3 = tf.concat([x3, i2], axis=-1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x8)
        x11 = tf.concat([x11, i3], axis=-1)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        return [x2, x10, x18]


class YOLOv4(tf.keras.models.Model):

    def __init__(self, num_classes=1, num_anchors=5, filters=1024):
        super().__init__()
        self.bb_0 = BackboneBase(filters // 16)
        self.bb_1 = BackboneBlock(filters // 8, num_res_blocks=2)
        self.bb_2 = BackboneBlock(filters // 4, num_res_blocks=8)
        self.bb_3 = BackboneBlock(filters // 2, num_res_blocks=8)
        self.bb_4 = BackboneBlock(filters, num_res_blocks=4)
        self.neck = Neck(filters)
        self.head = Head(filters, num_classes, num_anchors)

    def build(self, image_size):
        input_shape = (None, *image_size, 3)
        super().build(input_shape=input_shape)

    def call(self, x, training=None, mask=None):
        d0 = self.bb_0(x)
        d1 = self.bb_1(d0)
        d2 = self.bb_2(d1)
        d3 = self.bb_3(d2)
        d4 = self.bb_4(d3)
        x20, x13, x6 = self.neck([d4, d3, d2])
        output = self.head([x20, x13, x6])
        return output

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        print(x.shape, y.shape, y_pred.shape)
        return 0


def create_model(args):
    model = YOLOv4()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
    model.compile(optimizer, metrics=metrics)
    return model


def evaluate_model(model, subset, steps):
    predictions = model.predict(subset, steps=steps, verbose=1)
