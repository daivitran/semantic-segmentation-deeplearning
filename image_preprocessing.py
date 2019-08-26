#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:34:19 2019

@author: david
"""

import cv2
import numpy as np
import math

IMG_SIZE = 256

def preprocess_image_array(image_array, attribute, method = "normalization"):
    """
    This function preprocesses an image. Specifically, it adds paddings to an image to make it square
    , then resizes it to 256 x 256, and finally applies scaling.
        
    Args:
        attribute: attribute of this image (either "data" or "label")
        image: an image
        
    Returns:
        an array representation of the square image
    """
    
    padded_image_array = pad_image_array(image_array)
    resized_image_array = resize_image_array(padded_image_array, IMG_SIZE, IMG_SIZE)
    scaled_image_array = resized_image_array
    if attribute == "data":
        scaled_image_array = scale_image_array(resized_image_array, method)
    return scaled_image_array
    
def pad_image_array(image_array):
    """
    This function adds paddings to an array representation of the image to make it square.
    Specifically, it adds an equal amount of paddings to each side of the shorter dimension
    so that it is equal to the bigger dimension.
    
    Args:
        image_array: an array representation of the image.
    
    Returns:
        an array representation of the square image.
    """
    
    x = image_array.shape[0]
    y = image_array.shape[1]
    
    if x == y:
        return image_array
    elif x > y:
        column = int((x - y) / 2)
        bigger_dimension = x
        min_pixel = np.min(image_array)
        padded_image_array = np.full((bigger_dimension, bigger_dimension), min_pixel)
        for i in range(x):
            for j in range(y):
                padded_image_array[i, column + j] = image_array[i, j]
        return padded_image_array
    else:
        row = int((y - x) / 2)
        bigger_dimension = y
        min_pixel = np.min(image_array)
        padded_image_array = np.full((bigger_dimension, bigger_dimension), min_pixel)
        for i in range(x):
            for j in range(y):
                padded_image_array[row + i, j] = image_array[i, j]
        return padded_image_array
    
    
def resize_image_array(image_array, width, height):
    """
    This function resizes an array representation of the image to given width and height.
    
    Args:
        image_array: an array representation of the image.
        width: width to resize to.
        height: height to resize to.
        
    Returns:
        an array representation of the image with adjusted width and height.
    """
    
    resized_image_array = cv2.resize(image_array.astype('float32'), dsize=(width, height),
                                     interpolation=cv2.INTER_NEAREST)
    return resized_image_array

def scale_image_array(image_array, method):
    """
    This function scales the intensity of an array representation of the image
    with either normalization to [-1, 1] or standardization.
    
    Args:
        image_array: an array representation of the image.
        method: either "normalization" or "standardization"
        
    Returns:
        an array representation of the image with scaled intensity.
    """
    
    if method == "normalization":
        max_pixel = np.max(image_array)
        min_pixel = np.min(image_array)
        upper_bound = min_pixel + (max_pixel - min_pixel) * 90.0 / 100
        lower_bound = min_pixel + (max_pixel - min_pixel) * 2 / 100
        scaled_image_array = np.zeros(image_array.shape)
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                if image_array[i, j] > upper_bound:
                    scaled_image_array[i, j] = 1
                elif image_array[i, j] < lower_bound:
                    scaled_image_array[i, j] = -1
                else:
                    normalized = truncate(-1 + 2 * (image_array[i, j] - min_pixel) / (max_pixel - min_pixel), 5)
                    scaled_image_array[i, j] = normalized
        return scaled_image_array
    elif method == "standardization":
        max_pixel = np.max(image_array)
        min_pixel = np.min(image_array)
        upper_bound = min_pixel + (max_pixel - min_pixel) * 90.0 / 100
        lower_bound = min_pixel + (max_pixel - min_pixel) * 2 / 100
        scaled_image_array = np.zeros(image_array.shape)
        scaled_image_array[:] = image_array
        scaled_image_array[scaled_image_array < lower_bound] = lower_bound
        scaled_image_array[scaled_image_array > upper_bound] = upper_bound
        mean = np.mean(scaled_image_array)
        std = np.std(scaled_image_array)
        for i in range(scaled_image_array.shape[0]):
            for j in range(scaled_image_array.shape[1]):
                    standardized = truncate((scaled_image_array[i, j] - mean) / std, 5)
                    scaled_image_array[i, j] = standardized
        return scaled_image_array
    else:
        return image_array


def truncate(f, n):
    """
    This function rounds a number "f" to "n" decimal places.
    
    Args:
        f: number to be rounded.
        n: desired decimal places.
    
    Returns:
        a number with "n" decimal places.
    """
    
    return math.floor(f * 10 ** n) / 10 ** n
