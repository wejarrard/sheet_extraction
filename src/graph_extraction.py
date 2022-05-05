# GRAPH CROPS


# IMPORT ALL NECESSARY PACKAGES FOR CROP & READING
from __future__ import print_function
from PIL import Image
import cv2
import numpy as np
import os
import numpy.ma as ma
import shutil
from openpyxl import Workbook
import numpy.ma as ma
#import imutils
from itertools import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
#import psycopg2
import os
import imaplib
import email
import time
import shutil
#from Crypto.PublicKey import RSA
#from Crypto.Cipher import AES, PKCS1_OAEP

from tqdm import tqdm
from skimage.io import imread, imshow, imsave

import math
from skimage.io import imread, imsave
from skimage.transform import resize

from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Convolution2D, BatchNormalization
from tensorflow.python.keras.layers.core import Dropout, Lambda
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras import backend as K
#from tensorflow.python.keras.optimizers import adam

import tensorflow as tf
from tensorflow import keras

import glob

def alignImages(im1, im2, MAX_FEATURES, GOOD_MATCH_PERCENT):
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h



def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def crop_intraop_graph(image_path,output_path):
    refFilename = "IntraoperativeForm.JPG"
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    refFileWidth = imReference.shape[1]

    imFilename = image_path
    imPatientOrg = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    patient = os.path.basename(imFilename)[:-4]
    imPatient = image_resize(imPatientOrg, width=refFileWidth)


    # Registered image will be restored in img.
    # img = aligned image (cv2)
    # im = aligned image (PIL)
    # The estimated homography will be stored in h.
    img, h = alignImages(imPatient, imReference, MAX_FEATURES=500, GOOD_MATCH_PERCENT=0.18)
    im = Image.fromarray(img)


    # determine chart pixel area (on standard form)
    standImg = cv2.imread(refFilename)
    standIm = Image.open(refFilename)
    edges = cv2.Canny(standImg, 50, 110)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=300, lines=np.array([]), minLineLength=1, maxLineGap=50)

    # determine chart edges -> based on location in IntraoperativeRecord.jpg (or standard form)
    minX = lines[0][0][0]
    minY = lines[0][0][1]
    maxX = lines[0][0][0]
    maxY = lines[0][0][1]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x1 < minX):
            minX = x1
        if (x2 > maxX):
            maxX = x2
        if (y1 < minY):
            minY = y1
        if (y2 > maxY):
            maxY = y2

    Xdim = maxX - minX
    Ydim = maxY - minY

    edgeImg = img
    cv2.line(edgeImg, (minX, minY), (minX, maxY), (0, 0, 255), 2)
    cv2.line(edgeImg, (maxX, minY), (maxX, maxY), (0, 0, 255), 2)
    cv2.line(edgeImg, (minX, minY), (maxX, minY), (0, 0, 255), 2)
    cv2.line(edgeImg, (minX, maxY), (maxX, maxY), (0, 0, 255), 2)

    # Show Flowsheet Sections
    # SECTION 1
    XMin1 = minX + int((20 / 1190) * Xdim)
    XMax1 = minX + int((116 / 1190) * Xdim)
    YMin1 = minY + int((20 / 756) * Ydim)
    YMax1 = minY + int((310 / 756) * Ydim)

    # SECTION 2
    XMin2 = minX + int((120 / 1190) * Xdim)
    XMax2 = minX + int((1102 / 1190) * Xdim)
    YMin2 = minY + int((20 / 756) * Ydim)
    YMax2 = minY + int((310 / 756) * Ydim)

    # SECTION 3
    XMin3 = minX + int((1102 / 1190) * Xdim)
    XMax3 = minX + int((1190 / 1190) * Xdim)
    YMin3 = minY + int((20 / 756) * Ydim)
    YMax3 = minY + int((310 / 756) * Ydim)

    # SECTION 4
    XMin4 = minX + int((116 / 1190) * Xdim)
    XMax4 = minX + int((1102 / 1190) * Xdim)
    YMin4 = minY + int((310 / 756) * Ydim)
    YMax4 = minY + int((474 / 756) * Ydim)

    # SECTION 5
    XMin5 = minX + int((116 / 1190) * Xdim)
    XMax5 = minX + int((1102 / 1190) * Xdim)
    YMin5 = minY + int((474 / 756) * Ydim)
    YMax5 = minY + int((583 / 756) * Ydim)

    # SECTION 6
    XMin6 = minX + int((0 / 1190) * Xdim)
    XMax6 = minX + int((1190 / 1190) * Xdim)
    YMin6 = minY + int((587 / 756) * Ydim)
    YMax6 = minY + int((696 / 756) * Ydim)

    # SECTION 7
    XMin7 = minX + int((0 / 1190) * Xdim)
    XMax7 = minX + int((1190 / 1190) * Xdim)
    YMin7 = minY + int((701 / 756) * Ydim)
    YMax7 = minY + int((756 / 756) * Ydim)
    
    graph_crop = np.array(im.crop((XMin4, YMin4, XMax4+1, YMax4+1)))
    imsave(output_path + id_.split(".")[0] + '.png', graph_crop)    # save file for graph reading
    return graph_crop


# GRAPH READING
def read_in_graphs(dir_path,img_path):
    DIR_PATH = dir_path   # output path that was specified in previous function
    IMG_PATH = img_path    # to obtain specific graph crop
    img = imread(DIR_PATH + IMG_PATH)

    IMG_HEIGHT = 164 # Original Sizes
    IMG_WIDTH = 990
    NEW_HEIGHT = 256 # Sizes for Model
    NEW_WIDTH = 1024
    IMG_CHANNELS = 1 # We convert to grayscale
    batch_size = 1 # can only do batch_size = 1 with one image at a time
    pred_threshold = 0.5

    # Adjust image
    X = np.zeros([1, NEW_HEIGHT, NEW_WIDTH, IMG_CHANNELS])
    X[0,:IMG_HEIGHT,:IMG_WIDTH, 0] = np.expand_dims(255 * rgb2gray(img),0)

    # Black out extra writing
    graphEnd_model = load_model('GraphEnd_UNet.h5', compile = False)

    graphEnd_pred =  graphEnd_model.predict(X,batch_size = batch_size,verbose=1)
    graphEnd = (graphEnd_pred  > pred_threshold).astype(np.bool)
    del graphEnd_model

    graphEnd_boxes = np.zeros(X.shape)
    regions = regionprops(label(graphEnd[0,:,:,0]))
    maskFinal = np.zeros([NEW_HEIGHT, NEW_WIDTH])

    for region in regions:
        [x_min, y_min, x_max, y_max] = region.bbox
        mask = Image.new('L', (NEW_HEIGHT, NEW_WIDTH), 0)
        poly = np.zeros([8])
        poly[0] = x_min
        poly[1] = y_min
        poly[2] = x_min
        poly[3] = y_max
        poly[4] = NEW_HEIGHT
        poly[5] = y_max
        poly[6] = NEW_HEIGHT
        poly[7] = y_min
        poly = poly.tolist()
        ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
        mask = np.transpose(np.array(mask))
        maskFinal += mask
    graphEnd_boxes[0,:,:,0] = maskFinal
    graphEnd_boxes = graphEnd_boxes > 0

    graphEnd_boxes = np.squeeze(graphEnd_boxes)
    X_new = np.zeros([1, NEW_HEIGHT, NEW_WIDTH, IMG_CHANNELS], dtype=np.uint8)
    X_new[0,:,:,0] = X[0,:,:,0] * (1 - graphEnd_boxes)

    # Load systolic model
    systolic_model = load_model('Models/SBP_UNet.h5', compile = False)

    systolic_pred = systolic_model.predict(X_new, batch_size=batch_size, verbose=1)
    systolic = (systolic_pred > pred_threshold).astype(np.bool)
    del systolic_model

    # Load heart rate model
    hr_model = load_model('Heartrate_UNet.h5', compile = False)

    hr_pred = hr_model.predict(X_new, batch_size=batch_size, verbose=1)
    heart_rate = (hr_pred > pred_threshold).astype(np.bool)
    del hr_model

    # Load diastolic model
    diastolic_model = load_model('DBP_UNet.h5', compile = False)

    diastolic_pred = diastolic_model.predict(X_new, batch_size=batch_size, verbose=1)
    diastolic = (diastolic_pred > pred_threshold).astype(np.bool)
    del diastolic_model

    # Convert images back to normal
    X = np.squeeze(X_new[:,:IMG_HEIGHT,:IMG_WIDTH,:])
    systolic = np.squeeze(systolic[:,:IMG_HEIGHT,:IMG_WIDTH,:])
    heart_rate = np.squeeze(heart_rate[:,:IMG_HEIGHT,:IMG_WIDTH,:])
    diastolic = np.squeeze(diastolic[:,:IMG_HEIGHT,:IMG_WIDTH,:])


    # Post-processing to Time Series
    min_obj_size = 30
    pixel_interval = 16.5
    time_pixels = np.zeros(60)
    for i in range(60):
        if i > 0:
            if i % 2 == 0:
                time_pixels[i] = time_pixels[i-1] + 17
            else:
                time_pixels[i] = time_pixels[i-1] + 16
    time_pixels = time_pixels.astype(int)
    series_length = len(time_pixels)

    # Heart rate function
    def process_heartrates(heart_rate):
        hr_cleaned = label(remove_small_objects(heart_rate, 15))
        disk_el = disk(2)
        hr_cleaned = opening(hr_cleaned, disk_el)
        hr_props = regionprops(hr_cleaned)
        num_regions = len(hr_props)
        time_series = np.zeros([len(hr_props), 2])
        for i in range(num_regions):
            time_series[i, 0] = round(hr_props[i].centroid[1] / pixel_interval)
            time_series[i, 1] = IMG_HEIGHT - hr_props[i].centroid[0]
        hr_result = time_series[time_series[:,0].argsort()]
        ts_heartrate = np.zeros(series_length)
        for k in range(num_regions):
            ts_heartrate[hr_result[k,0].astype(int)] = round(hr_result[k,1],1)
        for k in range(series_length):
            if ts_heartrate[k] > 0:
                ts_heartrate[k] = round((ts_heartrate[k] - 13) / (IMG_HEIGHT - 13) * (210 - 30) + 30, 0)
        return ts_heartrate

    # Blood pressure function
    def process_bloodpressure(systolic, diastolic, bp_break):
        systolic_cleaned = remove_small_objects(systolic, min_obj_size)
        diastolic_cleaned = remove_small_objects(diastolic, min_obj_size)
        ts_systolic = np.zeros(series_length)
        ts_diastolic = np.zeros(series_length)
        step = 0
        zero_count = 0
        zero_threshold = 15
        activated = False  # Makes sure surgery has started before using BP break
        for t in time_pixels:

            # Systolic part
            for i in range(IMG_HEIGHT - 1, 0, -1):
                if systolic_cleaned[i, t] == 1:
                    ts_systolic[step] = IMG_HEIGHT - i
                    activated = True
                    break
            if bp_break == True:
                if ts_systolic[step] == 0:
                    zero_count += 1
                else:
                    zero_count = 0
                if zero_count == zero_threshold and activated == True:
                    break

                    # Diastolic part
            for j in range(IMG_HEIGHT):
                if diastolic_cleaned[j, t] == 1:
                    ts_diastolic[step] = IMG_HEIGHT - j
                    activated = True
                    break
            if bp_break == True:
                if ts_diastolic[step] == 0:
                    zero_count += 1
                else:
                    zero_count = 0
                if zero_count == zero_threshold and activated == True:
                    break

            step += 1

    for k in range(series_length):
        if ts_systolic[k] > 0:
            ts_systolic[k] = round((ts_systolic[k] - 13) / (IMG_HEIGHT - 13) * (210 - 30) + 30, 0) + 4
        if ts_diastolic[k] > 0:
            ts_diastolic[k] = round((ts_diastolic[k] - 13) / (IMG_HEIGHT - 13) * (210 - 30) + 30, 0) - 4
    return ts_systolic, ts_diastolic


    ts_heartrate = process_heartrates(heart_rate)
    ts_systolic, ts_diastolic = process_bloodpressure(systolic, diastolic, bp_break=True)

    ts_MAP = np.zeros(series_length, dtype=float)
    for i in range(series_length):
        if ts_systolic[i] > 0 and ts_diastolic[i] > 0:
            ts_MAP[i] = (2 * ts_diastolic[i] + ts_systolic[i]) / 3

    hr_detects = 5 * np.count_nonzero(ts_heartrate)
    dbp_detects = 5 * np.count_nonzero(ts_diastolic)
    sbp_detects = 5 * np.count_nonzero(ts_systolic)
    map_detects = 5 * np.count_nonzero(ts_MAP)

    hr_lesser_50 = 5 * np.count_nonzero((ts_heartrate < 50) & (ts_heartrate > 0))
    hr_greater_90 = 5 * np.count_nonzero((ts_heartrate > 90) & (ts_heartrate > 0))
    sbp_lesser_70 = 5 * np.count_nonzero((ts_systolic < 70) & (ts_systolic > 0))
    map_lesser_49 = 5 * np.count_nonzero((ts_MAP < 49) & (ts_MAP > 0))
    dbp_lesser_30 = 5 * np.count_nonzero((ts_diastolic < 30) & (ts_diastolic > 0))
    sbp_greater_160 = 5 * np.count_nonzero(ts_systolic > 160)
    sbp_between_140_160 = 5 * np.count_nonzero((ts_systolic >= 140) & (ts_systolic <= 160))
    sbp_between_80_90 = 5 * np.count_nonzero((ts_systolic >= 80) & (ts_systolic <= 90))
    sbp_lesser_80 = 5 * np.count_nonzero((ts_systolic < 80) & (ts_systolic > 0))
    dbp_between_50_60 = 5 * np.count_nonzero((ts_diastolic >= 50) & (ts_diastolic <= 60))
    dbp_lesser_50 = 5 * np.count_nonzero((ts_diastolic < 50) & (ts_diastolic > 0))

    results = pd.DataFrame()

    results['Name'] = [IMG_PATH]

    results['# HR'] = int(hr_detects / 5)
    results['# SBP'] = int(sbp_detects / 5)
    results['# DBP'] = int(dbp_detects / 5)
    results['# MAP'] = int(map_detects / 5)

    results['HR < 50'] = hr_lesser_50
    results['HR > 90'] = hr_greater_90
    results['SBP < 70'] = sbp_lesser_70
    results['MAP < 49'] = map_lesser_49
    results['DBP < 30'] = dbp_lesser_30
    results['SBP > 160'] = sbp_greater_160
    results['SBP 140-160'] = sbp_between_140_160
    results['SBP 80-90'] = sbp_between_80_90
    results['SBP < 80'] = sbp_lesser_80
    results['DBP 50-60'] = dbp_between_50_60
    results['DBP < 50'] = dbp_lesser_50

    if hr_detects > 0:
        results['% HR < 50'] = hr_lesser_50 / hr_detects
        results['% HR > 90'] = hr_greater_90 / hr_detects
    else:
        results['% HR < 50'] = None
        results['% HR > 90'] = None

    if sbp_detects > 0:
        results['% SBP < 70'] = sbp_lesser_70 / sbp_detects
        results['% SBP > 160'] = sbp_greater_160 / sbp_detects
        results['% SBP 140-160'] = sbp_between_140_160 / sbp_detects
        results['% SBP 80-90'] = sbp_between_80_90 / sbp_detects
        results['% SBP < 80'] = sbp_lesser_80 / sbp_detects
    else:
        results['% SBP < 70'] = None
        results['% SBP > 160'] = None
        results['% SBP 140-160'] = None
        results['% SBP 80-90'] = None
        results['% SBP < 80'] = None

    if map_detects > 0:
        results['% MAP < 49'] = map_lesser_49 / map_detects
    else:
        results['% MAP < 49'] = None

    if dbp_detects > 0:
        results['% DBP < 30'] = dbp_lesser_30 / dbp_detects
        results['% DBP 50-60'] = dbp_between_50_60 / dbp_detects
        results['% DBP < 50'] = dbp_lesser_50 / dbp_detects
    else:
        results['% DBP < 30'] = None
        results['% DBP 50-60'] = None
        results['% DBP < 50'] = None

    results = results.set_index('Name')
    return results
    


