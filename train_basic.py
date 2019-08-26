#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:42:32 2019

@author: david
"""

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, concatenate, Input, BatchNormalization
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras import models

import numpy as np
import matplotlib.pyplot as plt
import os
import functools

from image_augmentation import _augment
    
def unet(pretrained_weights = None, input_size = (256, 256, 1)):
    
    def encoder_block(input_tensor, num_filters):
        encoder = Conv2D(num_filters, (3, 3), padding = 'same')(input_tensor)
        encoder = BatchNormalization()(encoder)
        encoder = Activation('relu')(encoder)
        encoder = Conv2D(num_filters, (3, 3), padding = 'same')(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Activation('relu')(encoder)
        encoder_pool = MaxPooling2D((2, 2), strides = (2, 2))(encoder)
        return encoder_pool, encoder
    
    def center_block(input_tensor, num_filters):
        center = Conv2D(num_filters, (3, 3), padding = 'same')(input_tensor)
        center = Activation('relu')(center)
        center = Conv2D(num_filters, (3, 3), padding = 'same')(center)
        center = Activation('relu')(center)
        return center
        
    def decoder_block(input_tensor, concat_tensor, num_filters):
        decoder = Conv2DTranspose(num_filters, (2, 2), strides = (2, 2), padding = 'same')(input_tensor)
        decoder = concatenate([concat_tensor, decoder], axis = -1)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(num_filters, (3, 3), padding = 'same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(num_filters, (3, 3), padding = 'same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        return decoder

    inputs = Input(shape=input_size)
    # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    # 8
    center = center_block(encoder4_pool, 1024)
    # center
    decoder4 = decoder_block(center, encoder4, 512)
    # 16
    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32
    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64
    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss = bce_dice_loss, metrics = [dice_loss])
    
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model
    
def dice_coeff(y_true, y_pred):
    smooth = 1
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

# SETUP

TRAIN = os.path.join(os.path.abspath(''), "data/train.tfrecords")
VAL = os.path.join(os.path.abspath(''), "data/val.tfrecords")
TEST = os.path.join(os.path.abspath(''), "data/test.tfrecords")

num_trains = 44364
num_vals = 9506

train_batch_size = 15
val_batch_size = 15

epochs = 50

train_cfg = {
        "method": "normalization",
}

val_cfg = {
        "method": "normalization",
}

test_cfg = {
    "method": "normalization",
}

train_augment = functools.partial(_augment, **train_cfg)
val_augment = functools.partial(_augment, **val_cfg)
test_augment = functools.partial(_augment, **test_cfg)

def dataset_input_fn(file_path,
                     train,
                     batch_size,
                     buffer_size,
                     threads,
                     augment = functools.partial(_augment)):
    
    """
        This function reads tfrecords file into tensor and applies augmentation.
        
        Note:
            Make sure the images are resized into 256 x 256.
        
        Args:
            file_path: full path of the tfrecords file
            train: is this data used for training ?
            batch_size: how many data you want to load as once.
            threads: how many threads do you want to use.
            augment: augmentation method with other parameters except image and label
            are given.
        
        Returns:
            Dataset with batch_size x and y.
    """
    
    dataset = tf.data.TFRecordDataset(file_path)
    
    def parse(serialized):
        features = {
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'label': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'shape0': tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'shape1': tf.FixedLenFeature(shape=[], dtype=tf.int64)
            }
        
        example = tf.parse_single_example(serialized, features)
        image_raw = tf.decode_raw(example['image'], tf.float32)
        label_raw = tf.decode_raw(example['label'], tf.int64)
        shape0 = example['shape0']
        shape1 = example['shape1']
        image = tf.reshape(image_raw, tf.stack([shape0, shape1, 1]))
        label = tf.reshape(label_raw, tf.stack([shape0, shape1, 1]))
        return image, label
    
    dataset = dataset.map(parse, num_parallel_calls = threads)
    
    if train:
        dataset.map(augment, num_parallel_calls = threads)
        dataset = dataset.shuffle(buffer_size = buffer_size)

    dataset = dataset.repeat(None)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset

def train_input_fn():
    return dataset_input_fn(TRAIN, 
                            train = True, 
                            batch_size = train_batch_size, 
                            buffer_size = 6000,
                            threads = 4,
                            augment = train_augment)
    
def val_input_fn():
    return dataset_input_fn(VAL, 
                            train = False, 
                            batch_size = val_batch_size, 
                            buffer_size = 1,
                            threads = 4,
                            augment = val_augment)

# PATH
    
save_model_path = os.path.join(os.path.abspath(''), "basic_weights.hdf5")


# TRAINING (Comment this portion when testing)

model = unet()
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)
history = model.fit(train_input_fn(),
                    steps_per_epoch = int(np.ceil(num_trains / float(train_batch_size))),
                    epochs = epochs,
                    validation_data = val_input_fn(),
                    validation_steps = int(np.ceil(num_vals / float(val_batch_size))),
                    callbacks = [cp])

"""
# DISPLAY RESULT

dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, dice, label='Training Dice Loss')
plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig("basic_graph.png")
plt.show()

model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                           'dice_loss': dice_loss})
scores = model.evaluate(test_input_fn(), verbose=1)

text_file = open("Result.txt", "w")
text_file.write("bce dice loss: %s" % scores[0])
text_file.write("dice loss: %s" % scores[1])
text_file.close()
"""
