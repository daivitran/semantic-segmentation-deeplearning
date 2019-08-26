#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:56:23 2019

@author: david
"""

import SimpleITK as sitk
import os
import numpy as np
import tensorflow as tf
import sys
import functools

from image_preprocessing import preprocess_image_array
from image_augmentation import _augment

"""
Example at the bottom of the file.
"""

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
                'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                'label': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                'shape0': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
                'shape1': tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
                }
        
        example = tf.io.parse_single_example(serialized, features)
        image_raw = tf.io.decode_raw(example['image'], tf.float32)
        label_raw = tf.io.decode_raw(example['label'], tf.int64)
        shape0 = example['shape0']
        shape1 = example['shape1']
        image = tf.reshape(image_raw, (shape0, shape1, 1))
        label = tf.reshape(label_raw, (shape0, shape1, 1))
        return image, label

    dataset = dataset.map(parse, num_parallel_calls = threads)
    
    if train:
        dataset.map(augment, num_parallel_calls = threads)
        dataset = dataset.shuffle(buffer_size = buffer_size)
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.repeat(1)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def tfrecords_converter(data_folders, label_folders, out_paths, split_ratio):
    """
    This function converts different folders that contains data and label into
    tfrecords train data, tfrecords validation data and tfrecords test data.
    
    Note:
        Make sure there are no other files or hidden files rather than data.
        Make sure the names of data and the names of labels in a data folder
        and a label folder can be sorted equivalently e.g data00005.nii and
        label00005.nii have the same index value with respect to its folder.
    
    Args:
        data_folders: a list of paths of data folders. 
        
        label_folders: a list of paths of label folders.
        
        out_paths: a list of names of output tfrecords files e.g 
        train.tfrecords, val.tfrecords, test.tfrecords
        
        split_ratio: how to split the dataset e.g [0.8, 0.1, 0.1]
    
    Returns:
        four arrays of test image, test label, corresponding name for image, corresponding name for label.
    """
    
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    assert len(data_folders) == len(label_folders)
    
    total = 0
    count = 0 
    for i in range(len(data_folders)):
        assert len(os.listdir(data_folders[i])) == len(os.listdir(label_folders[i]))
        total = total + len(os.listdir(data_folders[i]))
    
    train_writer = tf.compat.v1.python_io.TFRecordWriter(out_paths[0])
    val_writer = tf.compat.v1.python_io.TFRecordWriter(out_paths[1])
    test_writer = tf.compat.v1.python_io.TFRecordWriter(out_paths[2])
    string_test_data = []
    string_test_label = []
    test_data = []
    test_label = []
    for i in range(len(data_folders)):
        data_folder = os.listdir(data_folders[i])
        label_folder = os.listdir(label_folders[i])
        data_folder.sort()
        label_folder.sort()
        size = len(data_folder)
        indices = list(range(size))
        np.random.shuffle(indices)
        
        train_size = int(split_ratio[0] * size)
        val_size = int(split_ratio[1] * size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        for j in train_indices:
            print_progress(count = count, total = total - 1)
            data_image = load_image(os.path.join(data_folders[i], data_folder[j]))
            label_image = load_image(os.path.join(label_folders[i], label_folder[j]))
            data_image_array = sitk.GetArrayFromImage(data_image)
            label_image_array = sitk.GetArrayFromImage(label_image)
            processed_data_image_array = preprocess_image_array(data_image_array, "data", "")
            processed_label_image_array = preprocess_image_array(label_image_array, "label", "")
            img = processed_data_image_array.astype(np.float32)
            lbl = processed_label_image_array.astype(np.int64)
            shape0 = np.int64(img.shape[0])
            shape1 = np.int64(img.shape[1])
            example = tf.train.Example(features = tf.train.Features(
                    feature = {
                            'image': _bytes_feature(img.tostring()),
                            'label': _bytes_feature(lbl.tostring()),
                            'shape0': _int64_feature(shape0),
                            'shape1': _int64_feature(shape1)
                            }))
            train_writer.write(example.SerializeToString())
            count = count + 1
        
        for j in val_indices:
            print_progress(count = count, total = total - 1)
            data_image = load_image(os.path.join(data_folders[i], data_folder[j]))
            label_image = load_image(os.path.join(label_folders[i], label_folder[j]))
            data_image_array = sitk.GetArrayFromImage(data_image)
            label_image_array = sitk.GetArrayFromImage(label_image)
            processed_data_image_array = preprocess_image_array(data_image_array, "data", "")
            processed_label_image_array = preprocess_image_array(label_image_array, "label", "")
            img = processed_data_image_array.astype(np.float32)
            lbl = processed_label_image_array.astype(np.int64)
            shape0 = np.int64(img.shape[0])
            shape1 = np.int64(img.shape[1])
            example = tf.train.Example(features = tf.train.Features(
                    feature = {
                            'image': _bytes_feature(img.tostring()),
                            'label': _bytes_feature(lbl.tostring()),
                            'shape0': _int64_feature(shape0),
                            'shape1': _int64_feature(shape1)
                            }))
            val_writer.write(example.SerializeToString())
            count = count + 1
            
        for j in test_indices:
            print_progress(count = count, total = total - 1)
            string_test_data.append(data_folder[j])
            string_test_label.append(label_folder[i])
            data_image = load_image(os.path.join(data_folders[i], data_folder[j]))
            label_image = load_image(os.path.join(label_folders[i], label_folder[j]))
            data_image_array = sitk.GetArrayFromImage(data_image)
            label_image_array = sitk.GetArrayFromImage(label_image)
            processed_data_image_array = preprocess_image_array(data_image_array, "data", "")
            processed_label_image_array = preprocess_image_array(label_image_array, "label", "")
            img = processed_data_image_array.astype(np.float32)
            lbl = processed_label_image_array.astype(np.int64)
            test_data.append(img)
            test_label.append(lbl)
            shape0 = np.int64(img.shape[0])
            shape1 = np.int64(img.shape[1])            
            example = tf.train.Example(features = tf.train.Features(
                    feature = {
                            'image': _bytes_feature(img.tostring()),
                            'label': _bytes_feature(lbl.tostring()),
                            'shape0': _int64_feature(shape0),
                            'shape1': _int64_feature(shape1)
                            }))
            test_writer.write(example.SerializeToString())
            count = count + 1
            
    train_writer.close()
    val_writer.close()
    test_writer.close()
    return np.array(test_data), np.array(test_label), np.array(string_test_data), np.array(string_test_label)
    
def load_image(path):
    """
    This function reads an image from a path.
    
    Args:
        path: path of an image
        
    Returns:
        An image is returned
    """
    
    image = sitk.ReadImage(path)
    return image

def print_progress(count, total):
    """
    This function prints progress of a task that involves iterations
    
    Args:
        count: how much work have been done
        total: total work needs to be done
        
    Returns:
        Nothing. Display progress bar on console.
    """
    
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

"""
SHADDEN_IMAGES = os.path.join(os.path.abspath(''), "shadden_images")
SHADDEN_LABELS = os.path.join(os.path.abspath(''), "shadden_labels")
CTISUS_IMAGES = os.path.join(os.path.abspath(''), "ctisus_images")
CTISUS_LABELS = os.path.join(os.path.abspath(''), "ctisus_labels")

data_folders = [SHADDEN_IMAGES, CTISUS_IMAGES]
label_folders = [SHADDEN_LABELS, CTISUS_LABELS]
split_ratio = [0.7, 0.15, 0.15]

TRAIN = os.path.join(os.path.abspath(''), "data/train.tfrecords")
VAL = os.path.join(os.path.abspath(''), "data/val.tfrecords")
TEST = os.path.join(os.path.abspath(''), "data/test.tfrecords")

out_paths = [TRAIN, VAL, TEST]

test_data, test_label, str_test_data, str_test_label = tfrecords_converter(data_folders, label_folders, out_paths, split_ratio)
np.save("test_data", test_data)
np.save("test_label", test_label)
np.save("str_test_data", str_test_data)
np.save("str_test_label", str_test_label)
"""
