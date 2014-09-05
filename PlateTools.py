import cv2
import numpy as np
import math
import sys
from scipy import ndimage
from scipy import misc
from PIL import ImageFilter
from PIL import Image

# DEFINE VARIABLES HERE.
MAX_PIXEL = 255

# FILTER_RATIO
# Given a matrix of edges, filter_ratio returns
# slice objects of connected components with the
# correct HEIGHT to WIDTH ratio (within epsilon difference).
# ARGUMENTS: edge matrix, Height to Width ratio (float), and epsilon (float)
# RETURNS: a LIST of SLICE OBJECTS that satisfy the ratio

def filter_ratio(edges, ratio, epsilon):
    # set up some variables!
    # ratio list contains all the candidates with the correct ratios
    ratio_list = []

    # label the connect components
    labeled_edges, num_labels = ndimage.label(edges)

    # FIRST, iterate through each component,
    # and check the ratio! if approx. 1:2, then add to list

    for label in range(1, num_labels + 1):
        # note: loc is a slice object
        loc = ndimage.find_objects(labeled_edges == label)[0]
        array = edges[loc]
        w1, h1 = len(array[0]), len(array)
        r = h1 / float(w1)
        if r > ratio - epsilon and r < ratio + epsilon:
            ratio_list.append(loc)

    return ratio_list


# THRESHOLD
# Given a matrix of pixel values, THRESHOLD
# 1) Retrieves tha average pixel value along the row of sample_height
# 2) Iterates through the data and marks as MAX_PIXEL/0 depending
# on above/below.
# 3) Keeps track of Total pixels along sample height
# ARGUMENTS: grayscale image (numpy matrix), and sample_height (int)
# RETURNS: thresholded image (numpy matrix), and number of pixels at sample height

def threshold(img, sample_h):
    # retreive properties
    img = img.copy()
    w, h = img.shape[1], img.shape[0]

    # average along sample_height
    avg = 0
    for x in range(0, w):
        avg += img.item((sample_h, x))
    avg /= w

    # Thresholding part!
    total = 0
    for x in range(0, w):
        for y in range(0, h):
            if img.item((y,x)) < avg:
                img.itemset((y,x), MAX_PIXEL)
                if y == sample_h:
                    total += MAX_PIXEL
            else:
                img.itemset((y,x), 0)

    return img, total


# COUNT_COMPS
# Given an image, COUNT_COMPS returns the
# number of unique connected components
# that lie along a sample_height.
# ARGUMENTS: image (matrix), and sample height (int)
# RETURNS: number of components (int)!

def count_comps(img, sample_height):
    comps, _ = ndimage.label(img)
    # converting to a set automatically removes duplicates
    return len(set(comps[sample_height]))


# SEGMENT
# Given an image, SEGMENT horizontally
# separates chunks of white components into parts.
# ARGUMENTS: image (matrix), min_width of chunk
# RETURNS: a LIST of TUPLES with begin and end x coordinates
# each of which represent one chunk.


def segment(img, min_width):
    # first, we create a "bar graph" that counts
    # the number of white pixels for a given x
    w, h = img.shape[1], img.shape[0]
    bar = np.zeros(w)
    for x in range(0, w):
        darkpix = 0
        for y in range(0, h):
            curr_pixel = img.item((y, x))
            if curr_pixel > MAX_PIXEL * .95:
                darkpix += 1
        bar.itemset(x, darkpix)

    # now, divide up the chunks into widths!
    begin = -1
    empty_thresh = 4
    widths = []

    length = len(bar)
    for z in range(0, length):
        if bar[z] <= empty_thresh and begin != -1 and z - begin > min_width:
            widths.append((begin, z))
            begin = -1
        elif bar[z] > empty_thresh and begin == -1:
            begin = z

    return widths


# SHARPEN
# Sharpens an image.
# ARGUMENT: Name of input file.
# OUTPUT: Name of output file.

def sharpen(img):

    OUTPUT = "sharpened.jpg"
    im = Image.open(img)
    im = im.filter(ImageFilter.SHARPEN)
    im.save(OUTPUT)

    return OUTPUT

