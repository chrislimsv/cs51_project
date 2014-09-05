import cv2
import numpy as np
import math
import sys
from scipy import ndimage
import EdgeData as edge
import PlateTools as pt
from PIL import Image
import pytesser


# CONSTANTS
PICTURE_MAX_LENGTH = 1000


###################################################
################     STAGE 0    ###################
#                 INITAL SET UP                   #
###################################################

# make sure user supplies a command line argument
if len(sys.argv) == 1:
    print "needs arguments"
    sys.exit(0)
# reads image, converts to grayscale
img_rgb = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
w, h = img.shape[0], img.shape[1]

if w > PICTURE_MAX_LENGTH or h > PICTURE_MAX_LENGTH:
    print ("ERROR: Width and height of picture must be less than "
        + str(PICTURE_MAX_LENGTH) + " pixels.")
    sys.exit(0)




###################################################
################     STAGE 1     ##################
#                EDGE DETECTION                   #
###################################################

print "APPLYING SOBEL FILTER..."
mags, dirs, max_mag = edge.sobel(img)

print "APPLYING NON-MAXIMUM SUPPRESSION..."
mags = edge.nonmax_supp(mags, dirs)

print "EDGE SMOOTHING USING HYSTERESIS..."
edges = edge.hysteresis(mags)


cv2.imwrite("edges.png", edges)



###################################################
################     STAGE 2     ##################
#            LICNSE PLATE LOCALIZATION            #
###################################################

print "LICENSE PLATE LOCALIZATION..."
print ">> TRIAL #1: NO DILATION"
trials = 3
for j in range(1, trials + 1):


    # all results will be stored in this list
    results = []

    # retrieve all edges with the correct ratios and small epsilon
    ratio_list = pt.filter_ratio(edges, 0.5, 0.125)

    # All license plates must have at least MIN_SIZE
    MIN_SIZE = 1600

    # count the number of components along the middle horizontal line!
    i = 0
    for loc in ratio_list:
        # target that satisfies ratio
        target = img[loc]
        target_w, target_h = target.shape[1], target.shape[0]
        middle_h = target_h / 2

        # if min_size is not satisfied, skip
        if target.size < MIN_SIZE:
            continue

        # threshold the middle thirds height of the image
        # and the middle half of width of image
        middle_third = target[(target_h/3):(target_h/3*2),(target_w/8):(target_w/8*7)]
        middle_third, _ = pt.threshold(middle_third, middle_third.shape[0] / 2)

        num = pt.count_comps(middle_third, middle_third.shape[0] / 2)

        cv2.imwrite("candidate" + str(i) + ".png", middle_third)
        i += 1

        # LICENSE PLATES must have more than 3 and less than 8 components along middle.
        if num > 3 and num < 9:
            results.append(loc)

    # Look at the number of results!
    # and if none found, move no further
    num_results = len(results)
    if num_results == 0:
        print "NO PLATES FOUND :("
        if j != trials - 1:
            print ">> TRIAL #" + str(j+1) + ": WITH DILATION"
            print ">>>> PERFORMING EDGE DILATION..."
            edges = edge.edge_dilation(edges, 1)
            cv2.imwrite("edges.png", edges)
        else:
            sys.exit(0)
    else:
        print str(num_results) + " PLATE(S) FOUND!"
        break




###################################################
################     STAGE 3     ##################
#         PROCESS PLATES / PREPARE FOR OCR        #
###################################################

print "LICENSE PLATE ANALYSIS..."

# for each plate, find where the characters are located!
# add processed versions to this following list:
processed = []

for loc in results:
    # First, retreive region and SHARPEN
    # to use sharpening library, we need to save, sharpen, and recall
    # Tedious, but necessary
    target = img[loc]
    cv2.imwrite("target.png", target)
    output = pt.sharpen("target.png")
    target = cv2.imread(output)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Retreive properties
    target_w, target_h = target.shape[1], target.shape[0]
    middle_h = target_h / 2

    # next, threshold along the middle line
    thresh, total = pt.threshold(target, middle_h)

    # We now "chop off" the top and bottom components
    # of the license plate. This is done by starting at
    # the middle of the plate, and moving upwards/downards
    # until arriving at a significant variation.

    up_bound = 0
    down_bound = 0

    # Move up until find in large variation
    for y in range(middle_h, target_h):
        row = 0
        for x in range(0, target_w):
            row += thresh.item((y, x))
        if row < total * .3 or row > total * 1.7:
            up_bound = y
            break

    # Now move down!
    reversed_range = reversed(range(0, middle_h))
    for y in reversed_range:
        row = 0
        for x in range(0, target_w):
            row += thresh.item((y, x))
        if row < total * .3 or row > total * 1.7:
            down_bound = y
            break

    # We have the up/down bounds. Cut image!
    cut_image = thresh[down_bound:up_bound]

    # if image is size 0, skip.
    if cut_image.size == 0:
        continue



    # Now we subject the cut component to
    # numerous restrictions. They are as follows:
    # 1) SIZE OF LETTER must be greater than MIN_SIZE.
    # 2) HEIGHT OF LETTER must be greater than MIN_HEIGHT.
    # 3) WIDTH OF LETTER must be greater than MIN_WIDTH;
    #    but to accomodate for I's, we allow small widths
    #    that are mostly filled.
    # 4) Bars that fill up the entire plate at left/right
    #    edges tend to show up; these are removed.


    # Variables for these restrictions
    new_w, new_h = cut_image.shape[1], cut_image.shape[0]
    min_size = 30
    min_width = .05 * new_w
    min_height = .25 * new_h

    # LABEL and ITERATE through the connected components
    comps, num = ndimage.label(cut_image)
    for label in range(1, num + 1):

        # retreive potential letters
        section = ndimage.find_objects(comps == label)[0]
        array = cut_image[section]

        # IMPOSE RESTRICTIONS.
        # RESTRICTION 1
        if (array.size < min_size
                # RESTRICTION 2
                or array.shape[0] < min_height
                # RESTRICTION 3
                or (array.shape[1] < min_width and
                    np.count_nonzero(array) < .9 * array.size)
                # RESTRICTION 4
                or (array.shape[0] == new_h and
                        (comps[0][0] == label
                            or comps[0][new_w-1] == label))):
            # failed to pass, so remove
            cut_image[section] = 0

    # DONE PROCESSING
    processed.append(cut_image)


###################################################
################     STAGE 4     ##################
#         ISOLATE CHARACTERS / TESSERACT          #
###################################################

# a counter!
i = 1

# go through each processed plates
for target in processed:

    # result string will store the end result!
    result_string = ""

    # segment into individual letters!
    # segments must have min_width of 3
    min_width = .05 * target.shape[1]
    lst = pt.segment(target, min_width)
    cv2.imwrite("target.png", target)

    # for each letter, save
    for elm in lst:
        seg = target[:,elm[0]:elm[1]]
        cv2.imwrite("letter.png", seg)
        result_string += pytesser.image_to_string("letter.png", psm=10)


    # strip spaces and newlines
    result_string = result_string.replace(' ', '')
    result_string = result_string.replace('\r', '')
    result_string = result_string.replace('\n', '')

    # PRINT OUT RESULT STRING!
    print "RESULT #" + str(i) + " (with segments): " + result_string

    # Repeat WITHOUT segmenting letters, for comparison.
    result_string = pytesser.image_to_string("target.png", psm=7)
    print "RESULT #" + str(i) + " (without segments): " + result_string

    i += 1


# AND DONE!!! :)
print "********  FINISH  ********"









