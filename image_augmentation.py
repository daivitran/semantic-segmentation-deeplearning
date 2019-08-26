#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:25:01 2019

@author: david
@version: tensorflow 1.12
"""

import tensorflow as tf
import tensorflow.contrib as tfcontrib
import numpy as np

# Assume all images have this SHAPE because we cannot extract shape during running.
SHAPE = [256, 256, 1]

def _augment(image, label,
             sharpness = 0.0, std_bound = [0.0, 0.0], k_bound = [0.0, 0.0],
             bluriness = 0.0,
             pertubation = 0.0, small_scale_bound = [0.0, 0.0], shift_bound = [0.0, 0.0],
             brightness = 0.0, delta = 0.0,
             gamma = 0.0, gamma_bound = [0.0, 0.0],
             scaling = 0.0, scale_bound = [0.0, 0.0],
             shift = 0.0, width_shift_range = 0.0, height_shift_range = 0.0,
             flip = 0.0,
             rotate = 0.0, angle = 0.0,
             distortion = 0.0, sigma = 0.0, alpha_bound = [0.0, 0.0],
             method = "normalization",
             ):
    
    """
    This function applies Deep Stack Transformation into a pair of input image and label.
    Specifically, it removes the brightest, darkest pixels first and applies 10 transformations
    with random probability and magnitude. The function performs normalization or standardization
    at the end.
    
    Returns:
        Transformed image and label.
    """
    
    image = remove_brightest_darkest(image)
    
    if lucky(sharpness):
        image = sharpen_image(image, SHAPE, std_bound, k_bound)
        
    if lucky(bluriness):
        image = blur_image(image, SHAPE, std_bound)
        
    if lucky(pertubation):
        image = pertubate_image(image, small_scale_bound, shift_bound)
        
    if lucky(brightness):
        image = adjust_brightness(image, delta)
        
    if lucky(gamma):
        image = gamma_image(image, gamma_bound)
    
    if lucky(scaling):
        image = scale(image, scale_bound)
        
    if lucky(shift):
        image, label = shift_image(image, label, SHAPE, width_shift_range, height_shift_range)
        
    if lucky(flip):
        image, label = flip_image(image, label)
        
    if lucky(rotate):
        image, label = rotate_image(image, label, angle)

    if lucky(distortion):
        image, label = _apply_elastic_distortion(image, label, SHAPE, sigma, alpha_bound)

    if method == "normalization":
        image = normalize(image)
    elif method == "standardization":
        image = tf.image.per_image_standardization(image)
    else:
        raise ValueError('You forgot method for scaling image !')

    return image, label

def remove_brightest_darkest(image):
    def second_max_tensor(image):
        max_pixel = tf.reduce_max(image)
        min_pixel = tf.reduce_min(image)
        min_pixel_tensor = tf.fill(tf.shape(image), min_pixel)
        image = tf.where(tf.equal(image, max_pixel), min_pixel_tensor, image)
        second_max_pixel = tf.reduce_max(image)
        tensor = tf.fill(tf.shape(image), second_max_pixel)
        return second_max_pixel, tensor
    def second_min_tensor(image):
        max_pixel = tf.reduce_max(image)
        min_pixel = tf.reduce_min(image)
        max_pixel_tensor = tf.fill(tf.shape(image), max_pixel)
        image = tf.where(tf.equal(image, min_pixel), max_pixel_tensor, image)
        second_min_pixel = tf.reduce_min(image)
        tensor = tf.fill(tf.shape(image), second_min_pixel)
        return second_min_pixel, tensor        
        
    upper_bound, upper_bound_tensor = second_max_tensor(image)
    lower_bound, lower_bound_tensor = second_min_tensor(image)
    image = tf.where(tf.less(image, lower_bound), lower_bound_tensor, image)
    image = tf.where(tf.greater(image, upper_bound), upper_bound_tensor, image)
    upper_bound = lower_bound + (upper_bound - lower_bound) * tf.constant(85.0 / 100)
    lower_bound = lower_bound + (upper_bound - lower_bound) * tf.constant(2.0 / 100)
    upper_bound_tensor = tf.fill(tf.shape(image), upper_bound)
    lower_bound_tensor = tf.fill(tf.shape(image), lower_bound)
    image = tf.where(tf.less(image, lower_bound), lower_bound_tensor, image)
    image = tf.where(tf.greater(image, upper_bound), upper_bound_tensor, image)
    return image

def lucky(chance):
    prob = np.random.uniform()
    return prob < chance

def normalize(image):
    big1 = tf.subtract(image, tf.reduce_min(image))
    big2 = tf.subtract(tf.reduce_max(image), tf.reduce_min(image))
    a = tf.cast(tf.constant(-1), tf.float32)
    b = tf.multiply(tf.cast(tf.constant(2), tf.float32), big1 / big2)
    return tf.add(a, b)

def sharpen_image(image, shape, std_bound = [0.0, 0.0], k_bound = [0.0, 0.0]):
    std = tf.random_uniform([], std_bound[0], std_bound[1])
    smoothed = image[tf.newaxis, :, :, :]
    kernel = gaussian_kernel(2, 0.0, std)[:, :, tf.newaxis, tf.newaxis]
    smoothed = tf.nn.conv2d(smoothed, kernel, strides=[1, 1, 1, 1], padding="SAME")
    smoothed = tf.reshape(smoothed, shape)
    edge = tf.subtract(image, smoothed)
    k = tf.random_uniform([], k_bound[0], k_bound[1])
    image = tf.add(image, tf.multiply(edge, k))
    return image

def blur_image(image, shape, std_bound = [0.0, 0.0]): #0.25, 1.5    
    std = tf.random_uniform([], std_bound[0], std_bound[1])
    image = image[tf.newaxis, :, :, :]
    kernel = gaussian_kernel(2, 0.0, std)[:, :, tf.newaxis, tf.newaxis]
    image = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")
    image = tf.reshape(image, shape)
    return image

def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def pertubate_image(image, scale_bound = [0.0, 0.0], shift_bound = [0.0, 0.0]): # 0.9, 1.1; -0.1, 0.1
        scale = tf.random_uniform([], scale_bound[0], scale_bound[1])
        shift = tf.random_uniform([], shift_bound[0], shift_bound[1])
        shift = shift * (tf.reduce_max(image) - tf.reduce_min(image))
        image = image * scale + shift
        return image

def adjust_brightness(image, delta = 0.0): # 0.1
    k = tf.random_uniform([], -delta, delta)
    amount = tf.multiply(k, tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))
    image = tf.image.adjust_brightness(image, amount)
    return image

def gamma_image(image, gamma_bound = [0.0, 0.0]):
    gain = np.random.uniform(low = 0, high = 1, size=[1,])
    gamma = np.random.uniform(low = gamma_bound[0], high = gamma_bound[1], size=[1,])
    image = tf.image.adjust_gamma(image, gamma=gamma[0], gain=gain[0])
    return image

def scale(image, scale_bound = [0.0, 0.0]): # 0.4, 1.6
    scale = tf.random_uniform([], scale_bound[0], scale_bound[1])
    image = image * scale
    return image

def shift_image(image, label, shape, width_shift_range = 0.0, height_shift_range = 0.0):
    
    def shift(image, shape, horizontal, verticle):
        paddings = tf.constant([])
        target_height = tf.constant(shape[0])
        target_width = tf.constant(shape[1])
        offset_height = tf.constant(0)
        offset_width = tf.constant(0)
        zeros = tf.constant([0, 0])
        min_pixel = tf.reduce_min(image) # tf.constant(-1)
        if horizontal < 0:
            paddings = tf.stack([zeros, tf.stack([tf.constant(0), tf.constant(-horizontal)]), zeros])
            offset_width = tf.constant(-horizontal)
        elif horizontal > 0:
            paddings = tf.stack([zeros, tf.stack([tf.constant(horizontal), tf.constant(0)]), zeros])
            offset_width = 0
        
        if not horizontal == 0:
            image = tf.pad(image, paddings, "CONSTANT", constant_values = min_pixel)
        
        if verticle < 0:
            paddings = tf.stack([tf.stack([tf.constant(-verticle), tf.constant(0)]), zeros, zeros])
            offset_height = 0
        elif verticle > 0:
            paddings = tf.stack([tf.stack([tf.constant(0), tf.constant(verticle)]), zeros, zeros])
            offset_height = tf.constant(verticle)
    
        if not verticle == 0:   
            image = tf.pad(image, paddings, "CONSTANT", constant_values = min_pixel)
        
        cropped = tf.image.crop_to_bounding_box(image,
                                                          offset_height,
                                                          offset_width,
                                                          target_height,
                                                          target_width)
        return cropped

    h = w = 0
    if width_shift_range:
        lower = -width_shift_range * shape[1]
        upper = width_shift_range * shape[1]
        w = np.random.randint(low = lower, high = upper)
    if height_shift_range:
        lower = -height_shift_range * shape[0]
        upper = height_shift_range * shape[0]
        h = np.random.randint(low = lower, high = upper)

    image, label = shift(image, shape, w, h), shift(label, shape, w, h)
    return image, label

def flip_image(image, label):
    
    def horizontal_flip(image, label):
        image, label = tf.image.flip_left_right(image), tf.image.flip_left_right(label)
        return image, label
    
    def verticle_flip(image, label):
        image, label = tf.image.flip_up_down(image), tf.image.flip_up_down(label)
        return image, label  
      
    verticle_or_horizontal = tf.random_uniform([], 0.0, 1.0)
    image, label = tf.cond(tf.less(verticle_or_horizontal, 0.5),
                           lambda: horizontal_flip(image, label),
                           lambda: verticle_flip(image, label))
    return image, label

def rotate_image(image, label, angle):
    angle = np.random.uniform(low = -angle, high = angle)
    image = tfcontrib.image.rotate(image, angles = angle)
    label = tfcontrib.image.rotate(image, angles = angle)
    return image, label

def _apply_elastic_distortion(image, label, shape, sigma, alpha_bound):

    def _create_elastic_distortion(shape, sigma, alpha):
        x = tf.random_uniform(shape, minval = -1, maxval = 1, dtype = tf.float32)
        y = tf.random_uniform(shape, minval = -1, maxval = 1, dtype = tf.float32)
        filter1 = gaussian_kernel(2, 0.0, sigma)
        filter2 = gaussian_kernel(2, 0.0, sigma)
        x = tf.expand_dims(x, 0)
        y = tf.expand_dims(y, 0)
        filter1 = tf.expand_dims(tf.expand_dims(filter1, len(shape) - 1), len(shape))
        filter2 = tf.expand_dims(tf.expand_dims(filter2, len(shape) - 1), len(shape))
        x = tf.nn.conv2d(x, filter1, strides = [1, 1, 1, 1], padding="SAME")
        y = tf.nn.conv2d(y, filter2, strides = [1, 1, 1, 1], padding="SAME")
        x = tf.reshape(x, shape[:-1])
        y = tf.reshape(y, shape[:-1])
        x = tf.multiply(x, tf.constant(alpha))
        y = tf.multiply(y, tf.constant(alpha))
        return x, y
        
    alpha = np.random.uniform(low = alpha_bound[0], high = alpha_bound[1])
    x, y = _create_elastic_distortion(shape, sigma, alpha)
    flow = tf.expand_dims(tf.stack([x, y], axis = len(shape) - 1), 0)
    img = tf.expand_dims(image, 0)
    img = tf.cast(img, tf.float32)
    lbl = tf.expand_dims(label, 0)
    lbl = tf.cast(lbl, tf.float32)
    img = tfcontrib.image.dense_image_warp(img, flow)
    lbl = tfcontrib.image.dense_image_warp(lbl, flow)
    lbl = tf.cast(lbl[0], tf.int64)
    return img, lbl
