import cv2
import numpy as np
import math
import sys
from scipy import ndimage


# DEFINE CONSTANTS HERE
MAX_PIXEL = 255


# SOBEL FILTER: Retrieve magnitudes and directions
# of gradients. Round directions to either 0, 45, 90, or 135.
# We save the max mag value to normalize our mags matrix later on.
# ARGUMENTS: grayscale image (in the form of numpy matrix)
# RETURNS: magnitude matrix, direction matrix, max magnitude

def sobel(img):

    # first, retreive some properties
    w, h = img.shape[0], img.shape[1]

    # mags will contain the magnitudes of gradients
    mags = img.copy()
    # dirs contain the ROUNDED angles of the gradient vectors
    dirs = np.zeros((w,h))

    mat_x = np.matrix('-1 0 1; -2 0 2; -1 0 1')
    mat_y = np.matrix('1 2 1; 0 0 0; -1 -2 -1')

    # for keeping track of maximum mag
    max_mag = 0

    steps = 0
    status = 0
    sys.stdout.write("0% ... ")
    sys.stdout.flush()
    for x in range(0, w):
        # print progress every 1/10 steps
        if steps == w / 10:
            status += 10
            if status == 100:
                print "100%"
            else:
                sys.stdout.write(str(status) + "% ... ")
                sys.stdout.flush()
            steps = 0

        for y in range(0, h):
            sum_x = 0
            sum_y = 0

            # edge cases
            if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                mags.itemset((x,y), 0)
                continue

            # Appli filter!!
            for n in range(-1, 2):
                for m in range(-1, 2):
                    sum_x += mat_x.item((n,m)) * img.item((x+n, y+m))
                    sum_y += mat_y.item((n,m)) * img.item((x+n, y+m))

            # compute magnitude of gradient
            mag = math.sqrt(sum_x ** 2 + sum_y ** 2)

            # if mag is greater than 255, truncate to 255
            # this means any mag greater than 255 is treated
            # equally (as a clear edge)
            if mag > 255:
                mag = 255

            # set pixel in result matrix
            mags.itemset((x,y), mag)

            # update max mag accordingly
            if mag > max_mag:
                max_mag = mag

            # Next: compute direction
            # but first check if sum_x is 0
            if sum_x == 0:
                theta = 90
            else:
                theta = math.degrees(math.atan2(sum_y, sum_x))
                # now we "round" the angles
                if ((theta >= -22.5 and theta < 22.5)
                        or theta > 157.5 or theta <= -157.5):
                    theta = 0
                elif ((theta >= 22.5 and theta < 67.5)
                        or (theta > -157.5 and theta <= -112.5)):
                    theta = 45
                elif ((theta >= 67.5 and theta < 112.5)
                        or (theta > -112.5 and theta <= -67.5)):
                    theta = 90
                else:
                    theta = 135

            # save the angle
            dirs.itemset((x,y), theta)
        steps += 1

    return mags, dirs, max_mag


# Non-maximum suppression. If pixel is smaller than
# left/right pixel (respective of edge direction), set its mag to 0.
# Arguments: Magnitude matrix and (rounded) direction matrix
# Returns: Magnitude matrix with suppression
def nonmax_supp(mags, dirs):
    # first, retreive properties!
    mags = mags.copy()
    w, h = mags.shape[0], mags.shape[1]

    for x in range(1, w - 1):
        for y in range(1, h - 1):
            theta = dirs.item((x,y))
            curr_pixel = mags.item((x,y))

            # compare theta!
            if theta == 0:
                # horizontal edge; compare up and down
                pixel1 = mags.item((x, y + 1))
                pixel2 = mags.item((x, y - 1))
            elif theta == 45:
                # compare southeast, northwest
                pixel1 = mags.item((x - 1, y - 1))
                pixel2 = mags.item((x + 1, y + 1))
            elif theta == 90:
                # vertical edge; compare left and right
                pixel1 = mags.item((x - 1, y))
                pixel2 = mags.item((x + 1, y))
            else:
                # compare northeast, southwest
                pixel1 = mags.item((x + 1, y + 1))
                pixel2 = mags.item((x - 1, y - 1))

            # now compare the pixel values
            # if left or right has a larger value, current pixel
            # cannot be an edge
            if (curr_pixel < pixel1 or curr_pixel < pixel2):
                mags.itemset((x,y), 0)

    return mags



# Hysteresis edge smoothing. Using high and low thresholds,
# we check which pixels are certain to be edges using the high
# threshold, and which pixels near the edges are also edges using
# the low threshold.
# ARGUMENTS: Matrix with edge data (0 for non-edge, MAX_PIXEL for edge)
# RETURNS: Modified edge data (with hysteresis applied)

def hysteresis(mags):
    # retreive properites
    w, h = mags.shape[0], mags.shape[1]

    thresh_high = 180
    thresh_low = 100
    edges = np.zeros((w,h))
    for x in range(0, w):
        for y in range(0, h):
            if (edges.item((x,y)) != MAX_PIXEL
                and mags.item((x,y)) >= thresh_high):
                edges.itemset((x,y), MAX_PIXEL)
                hystConnect(x, y, thresh_low, edges, mags)

    return edges


# Helper funtion for HYSTERESIS.
# Uses a stack, rather than recursion, to avoid stack overflow.
# hystConnect searches for pixels adjacent
# to definitive edges, and marks those above
# the LOW threshold as edges
def hystConnect(x, y, thresh, edges, mags):
    # retreive properties
    w, h = mags.shape[0], mags.shape[1]

    # stack instead of recursion!
    stack = [(x, y)]

    while len(stack) > 0:
        coords = stack.pop()
        x, y = coords[0], coords[1]

        for i in range(-1, 2):
            # edge cases for x!
            if (x == 0 and i == -1) or (x == w - 1 and i == 1):
                continue

            for j in range(-1, 2):
                # check edge cases for y
                if (y == 0 and j == -1) or (y == h - 1 and j == 1):
                    continue

                # if pixel is over threshold, mark as edge
                # and propagate
                # but if already marked as edge, ignore
                if (mags.item((x + i, y + j)) > thresh
                        and edges.item((x + i, y + j)) != MAX_PIXEL):
                    edges.itemset((x + i, y + j), MAX_PIXEL)
                    stack.append((x+i, y+j))




# Dilate the pixels!
# newly dilated pixels have value MAX_PIXEL - 1 (or 254)
# so that we don't dilate the newly added pixels!
# ARGUMENTS: Edge matrix, and number of times to operate dilation
# RETURNS: Matrix with dilated edges!

def edge_dilation(edges, num):
    # get properties
    edges = edges.copy()
    w, h = edges.shape[0], edges.shape[1]

    for i in range(1, num + 1):
        for x in range(0, w):
            for y in range(0, h):
                if edges.item((x,y)) >= MAX_PIXEL - i + 1:
                    if x != 0 and edges.item((x-1, y)) == 0:
                        edges.itemset((x-1, y), MAX_PIXEL - i)
                    elif y != 0 and edges.item((x, y-1)) == 0:
                        edges.itemset((x, y-1), MAX_PIXEL - i)
                    elif x != w - 1 and edges.item((x+1, y)) == 0:
                        edges.itemset((x+1, y), MAX_PIXEL - i)
                    elif y != h - 1 and edges.item((x, y+1)) == 0:
                        edges.itemset((x, y+1), MAX_PIXEL - i)
    return edges
