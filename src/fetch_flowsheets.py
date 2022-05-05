# Fetches unread emails from email (its somewhere)
from __future__ import print_function
from PIL import Image
import cv2
import numpy as np
import os
import numpy.ma as ma
import shutil
from openpyxl import Workbook
from dotenv import load_dotenv
import numpy.ma as ma
import imutils
from itertools import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imread
from datetime import datetime
import psycopg2
import os
import imaplib
import email
import time
import shutil
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP

######################################################################################################################

def CNN_function(image, name, patientID, flowsheetID):

    # CODE TO INTERPRET TEXT FIELDS
    results = 'results'
    # WRITE RESULTS TO DATABASE

def medicationID_function(image,name,patientID,flowsheetID):

    # CODE TO IDENTIFY MEDS
    results = 'results'
    # WRITE RESULTS TO DATABASE

def time_series_function(image,name,patientID,flowsheetID):

    # CODE TO INTERPRET TIME SERIES
    results = 'results'
    # WRITE RESULTS TO DATABASE

def digit_recognition_function(image,name,patientID,flowsheetID):

    # CODE TO INTERPRET NUMBERS
    results = 'results'
    # WRITE RESULTS TO DATABASE

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

def crop_intraop(image, patientID, flowsheetID):
    # Read reference image
    refFilename = "C:/Users/maryblankemeier/Desktop/IntegrationPackage/IntraoperativeForm.jpg"
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    refFileWidth = imReference.shape[1]

    imFilename = "C:/Users/maryblankemeier/Desktop/IntegrationPackage/" + image
    imPatientOrg = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    patient = os.path.basename(imFilename)[:-4]
    imPatient = image_resize(imPatientOrg, width=refFileWidth)

    # Registered image will be restored in img.
    # img = aligned image (cv2)
    # im = aligned image (PIL)
    # The estimated homography will be stored in h.
    img, h = alignImages(imPatient, imReference, MAX_FEATURES=500, GOOD_MATCH_PERCENT=0.1)
    im = Image.fromarray(img)


    # determine chart pixel area (on standard form)
    standImg = cv2.imread(refFilename)
    standIm = Image.open(refFilename)
    edges = cv2.Canny(standImg, 50, 110)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=300, lines=np.array([]), minLineLength=1,
                            maxLineGap=50)

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

    # SECTION 4 - Graph
    
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

    # SECTION 1:
    boxHeight1 = (YMax1 - YMin1) / 18
    top1 = YMin1
    bottom1 = YMin1 + boxHeight1

    for y in range(1, 19):

        if y!= 9 and y != 10 and y != 11:
            medicationID_function(np.array(im.crop((XMin1 - 2, top1 - 2, XMax1 + 2, bottom1 + 2))), 'Medication ' + str(y), patientID, flowsheetID)

        if y == 9:
            checkbox_function(np.array(im.crop((XMin1 - 2, top1 - 2, XMax1 + 2, bottom1 + 2))),
                              'Inhaled Drugs: Halothane', patientID, flowsheetID, 0)

        if y == 10:
            checkbox_function(np.array(im.crop((XMin1 - 2, top1 - 2, XMax1 + 2, bottom1 + 2))),
                              'Inhaled Drugs: Isoflurane', patientID, flowsheetID, 0)

        if y == 11:
            checkbox_function(np.array(im.crop((XMin1 - 2, top1 - 2, XMax1 + 2, bottom1 + 2))), 'Inhaled Drugs: Other',
                              patientID, flowsheetID, 0)

        top1 = bottom1
        bottom1 = top1 + boxHeight1

    # SECTION 2:
    boxWidth2 = (XMax2 - XMin2) / 60
    boxHeight2 = (YMax2 - YMin2) / 18

    top2 = YMin2
    bottom2 = top2 + boxHeight2

    for y in range(1, 19):
        time_series_function(np.array(im.crop((XMin2 - 2, top2 - 2, XMax2 + 2, bottom2 + 2))), 'Time Series ' + str(y), patientID, flowsheetID)

        top2 = bottom2
        bottom2 = top2 + boxHeight2

    # SECTION 3:
    boxHeight3 = (YMax3 - YMin3) / 18

    top3 = YMin3
    bottom3 = YMin3 + boxHeight3

    for y in range(1, 19):
        digit_recognition_function(np.array(im.crop((XMin3 - 2, top3 - 2, XMax3 + 2, bottom3 + 2))), 'Total ' + str(y), patientID, flowsheetID)

        top3 = bottom3
        bottom3 = top3 + boxHeight3

    # SECTION 4:
    graph_function(np.array(im.crop((XMin4, YMin4, XMax4, YMax4))), patientID, flowsheetID)


    # SECTION 5:
    boxHeight5 = (YMax5 - YMin5) / 7
    top5 = YMin5
    bottom5 = YMin5 + boxHeight5

    for y in range(1, 8):
        time_series_function(np.array(im.crop((XMin5 - 3, top5 - 3, XMax5 + 3, bottom5 + 3))), 'Time Series ' + str(y), patientID, flowsheetID)

        top5 = bottom5
        bottom5 = top5 + boxHeight5

    # SECTION 6:
    # Patient Safety
    checkbox_function(np.array(im.crop((XMin6 + ((5.5 / 385) * Xdim) - 2, YMin6 + ((10 / 241.5) * Ydim) - 2,
                                        XMin6 + ((30 / 385) * Xdim) + 2, YMin6 + ((12.5 / 241.5) * Ydim) + 2))),
                      'Patient Safety: EyeProtection', patientID, flowsheetID, 1)

    checkbox_function(np.array(im.crop((XMin6 + ((5.5 / 385) * Xdim) - 2, YMin6 + ((13.5 / 241.5) * Ydim) - 2,
                                        XMin6 + ((30 / 385) * Xdim) + 2, YMin6 + ((16 / 241.5) * Ydim) + 2))),
                      'Patient Safety: Warming', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((5.5 / 385) * Xdim) - 2, YMin6 + ((17 / 241.5) * Ydim) - 2, XMin6 + ((30 / 385) * Xdim) + 2,
                 YMin6 + ((19.5 / 241.5) * Ydim) + 2))), 'Patient Safety: TED Stockings', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((5.5 / 385) * Xdim) - 2, YMin6 + ((20.5 / 241.5) * Ydim) - 2, XMin6 + ((30 / 385) * Xdim) + 2,
                 YMin6 + ((23 / 241.5) * Ydim) + 2))), 'Patient Safety: Safety Checklist', patientID, flowsheetID, 1)

    # Mask Ventilation
    checkbox_function(np.array(
        im.crop((XMin6 + ((33.5 / 385) * Xdim) - 2, YMin6 + ((9 / 241.5) * Ydim) - 2, XMin6 + ((80 / 385) * Xdim) + 2,
                 YMin6 + ((12.5 / 241.5) * Ydim) + 2))), 'Mask Ventilation: Easy Ventilation', patientID, flowsheetID, 1)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((33.5 / 385) * Xdim) - 2, YMin6 + ((12.5 / 241.5) * Ydim) - 2, XMin6 + ((80 / 385) * Xdim) + 2,
         YMin6 + ((16 / 241.5) * Ydim) + 2))), 'Mask Ventilation: Ventilation w/ Adjunct', patientID, flowsheetID, 1)


    checkbox_function(np.array(
        im.crop((XMin6 + ((33.5 / 385) * Xdim) - 2, YMin6 + ((16 / 241.5) * Ydim) - 2, XMin6 + ((80 / 385) * Xdim) + 2,
                 YMin6 + ((19.5 / 241.5) * Ydim) + 2))), 'Mask Ventilation: Difficult Ventilation', patientID, flowsheetID, 1)


    # Airway
    checkbox_function(np.array(
        im.crop((XMin6 + ((84.5 / 385) * Xdim) - 2, YMin6 + ((10 / 241.5) * Ydim) - 2, XMin6 + ((130 / 385) * Xdim) + 2,
                 YMin6 + ((12.5 / 241.5) * Ydim) + 2))), 'Airway: Natural', patientID, flowsheetID, 1)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((84.5 / 385) * Xdim) - 2, YMin6 + ((13.5 / 241.5) * Ydim) - 2, XMin6 + ((130 / 385) * Xdim) + 2,
         YMin6 + ((16 / 241.5) * Ydim) + 2))), 'Airway: LMA', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((84.5 / 385) * Xdim) - 2, YMin6 + ((17 / 241.5) * Ydim) - 2, XMin6 + ((130 / 385) * Xdim) + 2,
                 YMin6 + ((19.5 / 241.5) * Ydim) + 2))), 'Airway: ETT', patientID, flowsheetID, 1)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((84.5 / 385) * Xdim) - 2, YMin6 + ((20.5 / 241.5) * Ydim) - 2, XMin6 + ((130 / 385) * Xdim) + 2,
         YMin6 + ((23 / 241.5) * Ydim) + 2))), 'Airway: Trach', patientID, flowsheetID, 1)

    # Airway Placement Aid
    checkbox_function(np.array(im.crop((XMin6 + ((165 / 385) * Xdim) - 2, YMin6 + ((21 / 756) * Ydim) - 2,
                                        XMin6 + ((175 / 385) * Xdim) + 2, YMin6 + ((34 / 756) * Ydim) + 2))),
                      'Airway Placement Aid: Used', patientID, flowsheetID, 1)
    checkbox_function(np.array(im.crop((XMin6 + ((177 / 385) * Xdim) - 2, YMin6 + ((21 / 756) * Ydim) - 2,
                                        XMin6 + ((195 / 385) * Xdim) + 2, YMin6 + ((34 / 756) * Ydim) + 2))),
                      'Airway Placement Aid: Not Used', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((134 / 385) * Xdim) - 2, YMin6 + ((31 / 756) * Ydim) - 2, XMin6 + ((170 / 385) * Xdim) + 2,
                 YMin6 + ((43 / 756) * Ydim) + 2))), 'Airway Placement Aid: Fibroscope', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((134 / 385) * Xdim) - 2, YMin6 + ((43 / 756) * Ydim) - 2, XMin6 + ((170 / 385) * Xdim) + 2,
                 YMin6 + ((54 / 756) * Ydim) + 2))), 'Airway Placement Aid: Bronchoscope', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((134 / 385) * Xdim) - 2, YMin6 + ((54 / 756) * Ydim) - 2, XMin6 + ((170 / 385) * Xdim) + 2,
                 YMin6 + ((65 / 756) * Ydim) + 1))), 'Airway Placement Aid: Other', patientID, flowsheetID, 1)

    # LRA
    checkbox_function(np.array(im.crop((XMin6 + ((207 / 385) * Xdim) - 2, YMin6 + ((6 / 241.5) * Ydim) - 2,
                                        XMin6 + ((218 / 385) * Xdim) + 2, YMin6 + ((11 / 241.5) * Ydim) + 2))),
                      'LRA: Used', patientID, flowsheetID, 1)

    checkbox_function(np.array(im.crop((XMin6 + ((219 / 385) * Xdim) - 2, YMin6 + ((6 / 241.5) * Ydim) - 2,
                                        XMin6 + ((235 / 385) * Xdim) + 2, YMin6 + ((11 / 241.5) * Ydim) + 2))),
                      'LRA: Not Used', patientID, flowsheetID, 1)

    # Tubes & Lines
    checkbox_function(np.array(
        im.crop((XMin6 + ((245 / 385) * Xdim) - 2, YMin6 + ((34 / 756) * Ydim) - 2, XMin6 + ((275 / 385) * Xdim) + 2,
                 YMin6 + ((46 / 756) * Ydim) + 2))), 'Tubes & Lines: Peripheral IV Line', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((245 / 385) * Xdim) - 2, YMin6 + ((46 / 756) * Ydim) - 2, XMin6 + ((275 / 385) * Xdim) + 2,
                 YMin6 + ((56 / 756) * Ydim) + 2))), 'Tubes & Lines: Central IV Line', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((245 / 385) * Xdim) - 2, YMin6 + ((56 / 756) * Ydim) - 2, XMin6 + ((275 / 385) * Xdim) + 2,
                 YMin6 + ((67 / 756) * Ydim) + 2))), 'Tubes & Lines: Urinary Catheter', patientID, flowsheetID, 1)

    checkbox_function(np.array(
        im.crop((XMin6 + ((245 / 385) * Xdim) - 2, YMin6 + ((67 / 756) * Ydim) - 2, XMin6 + ((275 / 385) * Xdim) + 2,
                 YMin6 + ((80 / 756) * Ydim) + 2))), 'Tubes & Lines: Gastric Tube', patientID, flowsheetID, 1)

    # Monitoring Details
    checkbox_function(np.array(
        im.crop((XMin6 + ((281 / 385) * Xdim) - 2, YMin6 + ((6.5 / 241.5) * Ydim) - 2, XMin6 + ((300 / 385) * Xdim) + 2,
                 YMin6 + ((9.5 / 241.5) * Ydim) + 2))), 'Monitoring Details: ECG', patientID, flowsheetID, 2)

    checkbox_function(np.array(
        im.crop((XMin6 + ((281 / 385) * Xdim) - 2, YMin6 + ((10 / 241.5) * Ydim) - 2, XMin6 + ((300 / 385) * Xdim) + 2,
                 YMin6 + ((13 / 241.5) * Ydim) + 2))), 'Monitoring Details: NIBP', patientID, flowsheetID, 2)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((281 / 385) * Xdim) - 2, YMin6 + ((13.5 / 241.5) * Ydim) - 2, XMin6 + ((300 / 385) * Xdim) + 2,
         YMin6 + ((16.5 / 241.5) * Ydim) + 2))), 'Monitoring Details: SpO2', patientID, flowsheetID, 2)

    checkbox_function(np.array(
        im.crop((XMin6 + ((281 / 385) * Xdim) - 2, YMin6 + ((17 / 241.5) * Ydim) - 2, XMin6 + ((300 / 385) * Xdim) + 2,
                 YMin6 + ((20 / 241.5) * Ydim) + 2))), 'Monitoring Details: EtCO2', patientID, flowsheetID, 2)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((281 / 385) * Xdim) - 2, YMin6 + ((20.5 / 241.5) * Ydim) - 2, XMin6 + ((300 / 385) * Xdim) + 2,
         YMin6 + ((23.5 / 241.5) * Ydim) + 2))), 'Monitoring Details: Stethoscope', patientID, flowsheetID, 2)

    checkbox_function(np.array(
        im.crop((XMin6 + ((304 / 385) * Xdim) - 2, YMin6 + ((6.5 / 241.5) * Ydim) - 2, XMin6 + ((330 / 385) * Xdim) + 2,
                 YMin6 + ((9.5 / 241.5) * Ydim) + 2))), 'Monitoring Details: Temperature', patientID, flowsheetID, 2)

    checkbox_function(np.array(
        im.crop((XMin6 + ((304 / 385) * Xdim) - 2, YMin6 + ((10 / 241.5) * Ydim) - 2, XMin6 + ((330 / 385) * Xdim) + 2,
                 YMin6 + ((13 / 241.5) * Ydim) + 2))), 'Monitoring Details: NMT', patientID, flowsheetID, 2)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((304 / 385) * Xdim) - 2, YMin6 + ((13.5 / 241.5) * Ydim) - 2, XMin6 + ((330 / 385) * Xdim) + 2,
         YMin6 + ((16.5 / 241.5) * Ydim) + 2))), 'Monitoring Details: Urine Output', patientID, flowsheetID, 2)

    checkbox_function(np.array(
        im.crop((XMin6 + ((304 / 385) * Xdim) - 2, YMin6 + ((17 / 241.5) * Ydim) - 2, XMin6 + ((330 / 385) * Xdim) + 2,
                 YMin6 + ((20 / 241.5) * Ydim) + 2))), 'Monitoring Details: Arterial BP', patientID, flowsheetID, 2)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((304 / 385) * Xdim) - 2, YMin6 + ((20.5 / 241.5) * Ydim) - 2, XMin6 + ((330 / 385) * Xdim) + 2,
         YMin6 + ((23.5 / 241.5) * Ydim) + 2))), 'Monitoring Details: Other', patientID, flowsheetID, 2)

    # Patient Position
    checkbox_function(np.array(im.crop(
        (XMin6 + ((336.5 / 385) * Xdim) - 2, YMin6 + ((6.5 / 241.5) * Ydim) - 2, XMin6 + ((355 / 385) * Xdim) + 2,
         YMin6 + ((9.5 / 241.5) * Ydim) + 2))), 'Patient Position: Supine', patientID, flowsheetID, 3)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((336.5 / 385) * Xdim) - 2, YMin6 + ((10 / 241.5) * Ydim) - 2, XMin6 + ((355 / 385) * Xdim) + 2,
         YMin6 + ((13 / 241.5) * Ydim) + 2))), 'Patient Position: Prone', patientID, flowsheetID, 3)

    checkbox_function(np.array(im.crop((XMin6 + ((336.5 / 385) * Xdim) - 2, YMin6 + ((13.5 / 241.5) * Ydim) - 2, XMin6 + ((355 / 385) * Xdim) + 2,
         YMin6 + ((16.5 / 241.5) * Ydim) + 2))), 'Patient Position: Lithotomy', patientID, flowsheetID, 3)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((336.5 / 385) * Xdim) - 2, YMin6 + ((17 / 241.5) * Ydim) - 2, XMin6 + ((355 / 385) * Xdim) + 2,
         YMin6 + ((20 / 241.5) * Ydim) + 2))), 'Patient Position: Sitting', patientID, flowsheetID, 3)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((359.5 / 385) * Xdim) - 2, YMin6 + ((6.5 / 241.5) * Ydim) - 2, XMin6 + ((380 / 385) * Xdim) + 2,
         YMin6 + ((9.5 / 241.5) * Ydim) + 2))), 'Patient Position: Trendelenburg', patientID, flowsheetID, 3)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((359.5 / 385) * Xdim) - 2, YMin6 + ((10 / 241.5) * Ydim) - 2, XMin6 + ((380 / 385) * Xdim) + 2,
         YMin6 + ((13 / 241.5) * Ydim) + 2))), 'Patient Position: Fowler', patientID, flowsheetID, 3)

    checkbox_function(np.array(im.crop(
        (XMin6 + ((359.5 / 385) * Xdim) - 2, YMin6 + ((13.5 / 241.5) * Ydim) - 2, XMin6 + ((380 / 385) * Xdim) + 2,
         YMin6 + ((16.5 / 241.5) * Ydim) + 2))), 'Patient Position: Lateral', patientID, flowsheetID, 3)

def crop_anesthesia(image, patientID, flowsheetID):
    # Read reference image
    refFilename = "C:/Users/maryblankemeier/Desktop/IntegrationPackage/AnesthesiaForm.jpg"
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    refFileWidth = imReference.shape[1]

    # Read image to be aligned
    imFilename = "C:/Users/maryblankemeier/Desktop/IntegrationPackage/" + image
    imPatientOrg = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    patient = os.path.basename(imFilename)[:-4]

    imPatient = image_resize(imPatientOrg, width=refFileWidth)

    # Registered image will be restored in img.
    # img = aligned image (cv2)
    # im = aligned image (PIL)
    # The estimated homography will be stored in h.
    img, h = alignImages(imPatient, imReference, MAX_FEATURES=1200, GOOD_MATCH_PERCENT=0.2)
    im = Image.fromarray(img)

    # determine chart pixel area (on standard form)
    standImg = cv2.imread(refFilename)
    standIm = Image.open(refFilename)
    edges = cv2.Canny(standImg, 50, 110)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=300, lines=np.array([]), minLineLength=1,
                            maxLineGap=50)

    # determine chart edges -> based on location in AnesthesiaRecord.jpg (or standard form)
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

    # Show Flowsheet Sections
    # Thyromental Distance
    XMin2 = minX
    XMax2 = minX + int((552 / 1196) * Xdim)
    YMin2 = minY + int((96 / 770) * Ydim)
    YMax2 = minY + int((156 / 770) * Ydim)

    # Mask Ventilation Evaluation
    XMin3 = minX
    XMax3 = minX + int((552 / 1196) * Xdim)
    YMin3 = minY + int((163 / 770) * Ydim)
    YMax3 = minY + int((195 / 770) * Ydim)

    # Relevant Lab Results
    XMin5 = minX
    XMax5 = minX + int((552 / 1196) * Xdim)
    YMin5 = minY + int((289 / 770) * Ydim)
    YMax5 = minY + int((349 / 770) * Ydim)

    # Relevant Study Results
    XMin6 = minX
    XMax6 = minX + int((552 / 1196) * Xdim)
    YMin6 = minY + int((357 / 770) * Ydim)
    YMax6 = minY + int((401 / 770) * Ydim)

    # ASA Classification
    XMin7 = minX
    XMax7 = minX + int((293 / 1196) * Xdim)
    YMin7 = minY + int((409 / 770) * Ydim)
    YMax7 = minY + int((501 / 770) * Ydim)

    # Consent Checklist
    XMin8 = minX + int((293 / 1196) * Xdim)
    XMax8 = minX + int((552 / 1196) * Xdim)
    YMin8 = minY + int((409 / 770) * Ydim)
    YMax8 = minY + int((501 / 770) * Ydim)

    # Planned Anesthesia
    XMin9 = minX
    XMax9 = minX + int((552 / 1196) * Xdim)
    YMin9 = minY + int((508 / 770) * Ydim)
    YMax9 = minY + int((660 / 770) * Ydim)

    # Pre-Op Assessment
    XMin10 = minX + int((645 / 1196) * Xdim)
    XMax10 = minX + int((1196 / 1196) * Xdim)
    YMin10 = minY + int((66 / 770) * Ydim)
    YMax10 = minY + int((122 / 770) * Ydim)

    # Allergies
    XMin11 = minX + int((645 / 1196) * Xdim)
    XMax11 = minX + int((1196 / 1196) * Xdim)
    YMin11 = minY + int((131 / 770) * Ydim)
    YMax11 = minY + int((179 / 770) * Ydim)

    # Past Medical History
    XMin12 = minX + int((645 / 1196) * Xdim)
    XMax12 = minX + int((1196 / 1196) * Xdim)
    YMin12 = minY + int((185 / 770) * Ydim)
    YMax12 = minY + int((262 / 770) * Ydim)

    # Surgical and Anesthetic History
    XMin13 = minX + int((645 / 1196) * Xdim)
    XMax13 = minX + int((1196 / 1196) * Xdim)
    YMin13 = minY + int((270 / 770) * Ydim)
    YMax13 = minY + int((468 / 770) * Ydim)

    # Current Medications
    XMin14 = minX + int((645 / 1196) * Xdim)
    XMax14 = minX + int((1196 / 1196) * Xdim)
    YMin14 = minY + int((474 / 770) * Ydim)
    YMax14 = minY + int((587 / 770) * Ydim)

    # Clinical Exam
    XMin15 = minX + int((645 / 1196) * Xdim)
    XMax15 = minX + int((1196 / 1196) * Xdim)
    YMin15 = minY + int((594 / 770) * Ydim)
    YMax15 = minY + int((770 / 770) * Ydim)

    # SECTION 2:
    # Section 2 Folder:
    checkbox_function(np.array(
        im.crop((XMin2 + ((138 / 1234) * Xdim) - 2, YMin2 + ((0 / 770) * Ydim) - 2, XMin2 + ((225 / 1234) * Xdim) + 2,
                 YMin2 + ((16 / 770) * Ydim) + 2))), 'Thyromental Distance: More than 6 cm', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((262 / 1234) * Xdim) - 2, YMin2 + ((0 / 770) * Ydim) - 2, XMin2 + ((341 / 1234) * Xdim) + 2,
                 YMin2 + ((16 / 770) * Ydim) + 2))), 'Thyromental Distance: 5 to 6 cm', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((414 / 1234) * Xdim) - 2, YMin2 + ((0 / 770) * Ydim) - 2, XMin2 + ((505 / 1234) * Xdim) + 2,
                 YMin2 + ((16 / 770) * Ydim) + 2))), 'Thyromental Distance: Less than 5 cm', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((138 / 1234) * Xdim) - 2, YMin2 + ((16 / 770) * Ydim) - 2, XMin2 + ((225 / 1234) * Xdim) + 2,
                 YMin2 + ((31 / 770) * Ydim) + 2))), 'Mouth Opening: 3 own Fingers', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((262 / 1234) * Xdim) - 2, YMin2 + ((16 / 770) * Ydim) - 2, XMin2 + ((341 / 1234) * Xdim) + 2,
                 YMin2 + ((31 / 770) * Ydim) + 2))), 'Mouth Opening: 2 own Fingers', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((414 / 1234) * Xdim) - 2, YMin2 + ((16 / 770) * Ydim) - 2, XMin2 + ((505 / 1234) * Xdim) + 2,
                 YMin2 + ((31 / 770) * Ydim) + 2))), 'Mouth Opening: 1 own Finger', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((138 / 1234) * Xdim) - 2, YMin2 + ((31 / 770) * Ydim) - 2, XMin2 + ((225 / 1234) * Xdim) + 2,
                 YMin2 + ((45 / 770) * Ydim) + 2))), 'Neck Mobility: Normal', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((262 / 1234) * Xdim) - 2, YMin2 + ((31 / 770) * Ydim) - 2, XMin2 + ((341 / 1234) * Xdim) + 2,
                 YMin2 + ((45 / 770) * Ydim) + 2))), 'Neck Mobility: Reduced', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((414 / 1234) * Xdim) - 2, YMin2 + ((31 / 770) * Ydim) - 2, XMin2 + ((505 / 1234) * Xdim) + 2,
                 YMin2 + ((45 / 770) * Ydim) + 2))), 'Neck Mobility: Blocked Flexion', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((138 / 1234) * Xdim) - 2, YMin2 + ((45 / 770) * Ydim) - 2, XMin2 + ((225 / 1234) * Xdim) + 2,
                 YMin2 + ((60 / 770) * Ydim) + 2))), 'Upper Incisors: Absent', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((262 / 1234) * Xdim) - 2, YMin2 + ((45 / 770) * Ydim) - 2, XMin2 + ((341 / 1234) * Xdim) + 2,
                 YMin2 + ((60 / 770) * Ydim) + 2))), 'Upper Incisors: Normal', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin2 + ((414 / 1234) * Xdim) - 2, YMin2 + ((45 / 770) * Ydim) - 2, XMin2 + ((505 / 1234) * Xdim) + 2,
                 YMin2 + ((60 / 770) * Ydim) + 2))), 'Upper Incisors: Prominent', patientID, flowsheetID, 4)

    # SECTION 3:
    checkbox_function(np.array(
        im.crop((XMin3 + ((3 / 1234) * Xdim) - 2, YMin3 + ((16 / 770) * Ydim) - 2, XMin3 + ((113 / 1234) * Xdim) + 2,
                 YMin3 + ((33 / 770) * Ydim) + 2))), 'Mask Ventilation Evaluation: Patient age > 55', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin3 + ((135 / 1234) * Xdim) - 2, YMin3 + ((16 / 770) * Ydim) - 2, XMin3 + ((230 / 1234) * Xdim) + 2,
                 YMin3 + ((33 / 770) * Ydim) + 2))), 'Mask Ventilation Evaluation: Patient has beard', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin3 + ((250 / 1234) * Xdim) - 2, YMin3 + ((16 / 770) * Ydim) - 2, XMin3 + ((350 / 1234) * Xdim) + 2,
                 YMin3 + ((33 / 770) * Ydim) + 2))), 'Mask Ventilation Evaluation: Patient has no teeth', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin3 + ((378 / 1234) * Xdim) - 2, YMin3 + ((16 / 770) * Ydim) - 2, XMin3 + ((460 / 1234) * Xdim) + 2,
                 YMin3 + ((33 / 770) * Ydim) + 2))), 'Mask Ventilation Evaluation: Patient snores', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin3 + ((484 / 1234) * Xdim) - 2, YMin3 + ((16 / 770) * Ydim) - 2, XMin3 + ((555 / 1234) * Xdim) + 2,
                 YMin3 + ((33 / 770) * Ydim) + 2))), 'Mask Ventilation Evaluation: BMI > 26', patientID, flowsheetID, 4)

    # SECTION 5:
    checkbox_function(np.array(im.crop((XMin5 + ((151 / 1234) * Xdim) - 2, YMin5 + ((0 / 770) * Ydim) - 2,
                                        XMin5 + ((231 / 1234) * Xdim) + 2, YMin5 + ((16 / 770) * Ydim) + 2))),
                      'Relevant Lab Results: Yes', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop((XMin5 + ((235 / 1234) * Xdim) - 2, YMin5 + ((0 / 770) * Ydim) - 2,
                                        XMin5 + ((275 / 1234) * Xdim) + 2, YMin5 + ((16 / 770) * Ydim) + 2))),
                      'Relevant Lab Results: No', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((2 / 1234) * Xdim) - 2, YMin5 + ((20 / 785) * Ydim) - 2, XMin5 + ((140 / 1234) * Xdim) + 2,
                 YMin5 + ((32 / 785) * Ydim) + 2))), 'Relevant Lab Results: HGB', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((2 / 1234) * Xdim) - 2, YMin5 + ((34 / 785) * Ydim) - 2, XMin5 + ((140 / 1234) * Xdim) + 2,
                 YMin5 + ((46 / 785) * Ydim) + 2))), 'Relevant Lab Results: HCT', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((2 / 1234) * Xdim) - 2, YMin5 + ((48 / 785) * Ydim) - 2, XMin5 + ((140 / 1234) * Xdim) + 2,
                 YMin5 + ((60 / 785) * Ydim) + 2))), 'Relevant Lab Results: PLT', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((146 / 1234) * Xdim) - 2, YMin5 + ((20 / 785) * Ydim) - 2, XMin5 + ((280 / 1234) * Xdim) + 2,
                 YMin5 + ((32 / 785) * Ydim) + 2))), 'Relevant Lab Results: aPTT', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((146 / 1234) * Xdim) - 2, YMin5 + ((34 / 785) * Ydim) - 2, XMin5 + ((280 / 1234) * Xdim) + 2,
                 YMin5 + ((46 / 785) * Ydim) + 2))), 'Relevant Lab Results: NR', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((146 / 1234) * Xdim) - 2, YMin5 + ((48 / 785) * Ydim) - 2, XMin5 + ((280 / 1234) * Xdim) + 2,
                 YMin5 + ((60 / 785) * Ydim) + 2))), 'Relevant Lab Results: Na', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((288 / 1234) * Xdim) - 2, YMin5 + ((20 / 785) * Ydim) - 2, XMin5 + ((425 / 1234) * Xdim) + 2,
                 YMin5 + ((32 / 785) * Ydim) + 2))), 'Relevant Lab Results: Cl', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((288 / 1234) * Xdim) - 2, YMin5 + ((34 / 785) * Ydim) - 2, XMin5 + ((425 / 1234) * Xdim) + 2,
                 YMin5 + ((46 / 785) * Ydim) + 2))), 'Relevant Lab Results: K', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((288 / 1234) * Xdim) - 2, YMin5 + ((48 / 785) * Ydim) - 2, XMin5 + ((425 / 1234) * Xdim) + 2,
                 YMin5 + ((60 / 785) * Ydim) + 2))), 'Relevant Lab Results: Creatinine', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((431 / 1234) * Xdim) - 2, YMin5 + ((20 / 785) * Ydim) - 2, XMin5 + ((570 / 1234) * Xdim) + 2,
                 YMin5 + ((32 / 785) * Ydim) + 2))), 'Relevant Lab Results: Urea', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((431 / 1234) * Xdim) - 2, YMin5 + ((34 / 785) * Ydim) - 2, XMin5 + ((570 / 1234) * Xdim) + 2,
                 YMin5 + ((46 / 785) * Ydim) + 2))), 'Relevant Lab Results: Glycemia', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin5 + ((431 / 1234) * Xdim) - 2, YMin5 + ((48 / 785) * Ydim) - 2, XMin5 + ((570 / 1234) * Xdim) + 2,
                 YMin5 + ((60 / 785) * Ydim) + 2))), 'Relevant Lab Results: Other', patientID, flowsheetID, 4)

    # SECTION 6:
    checkbox_function(np.array(
        im.crop((XMin6 + ((161 / 1234) * Xdim) - 2, YMin6 + ((0 / 770) * Ydim) - 2, XMin6 + ((240 / 1234) * Xdim) + 2,
                 YMin6 + ((16 / 770) * Ydim) + 2))), 'Relevant Study Results: Yes', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin6 + ((247 / 1234) * Xdim) - 2, YMin6 + ((0 / 770) * Ydim) - 2, XMin6 + ((280 / 1234) * Xdim) + 2,
                 YMin6 + ((16 / 770) * Ydim) + 2))), 'Relevant Study Results: No', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin6 + ((4 / 1234) * Xdim) - 2, YMin6 + ((19 / 785) * Ydim) - 2, XMin6 + ((280 / 1234) * Xdim) + 2,
                 YMin6 + ((31 / 785) * Ydim) + 2))), 'Relevant Study Results: Chest Xray', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin6 + ((4 / 1234) * Xdim) - 2, YMin6 + ((31 / 785) * Ydim) - 2, XMin6 + ((280 / 1234) * Xdim) + 2,
                 YMin6 + ((44 / 785) * Ydim) + 2))), 'Relevant Study Results: ECG', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin6 + ((293 / 1234) * Xdim) - 2, YMin6 + ((19 / 785) * Ydim) - 2, XMin6 + ((560 / 1234) * Xdim) + 2,
                 YMin6 + ((31 / 785) * Ydim) + 2))), 'Relevant Study Results: Other1', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin6 + ((293 / 1234) * Xdim) - 2, YMin6 + ((31 / 785) * Ydim) - 2, XMin6 + ((560 / 1234) * Xdim) + 2,
                 YMin6 + ((44 / 785) * Ydim) + 2))), 'Relevant Study Results: Other2', patientID, flowsheetID, 4)

    # SECTION 7:
    checkbox_function(np.array(
        im.crop((XMin7 + ((7 / 1234) * Xdim) - 2, YMin7 + ((19 / 785) * Ydim) - 2, XMin7 + ((302 / 1234) * Xdim) + 2,
                 YMin7 + ((31 / 785) * Ydim) + 2))), 'ASA Classification: 1', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin7 + ((7 / 1234) * Xdim) - 2, YMin7 + ((29 / 785) * Ydim) - 2, XMin7 + ((302 / 1234) * Xdim) + 2,
                 YMin7 + ((41 / 785) * Ydim) + 2))), 'ASA Classification: 2', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin7 + ((7 / 1234) * Xdim) - 2, YMin7 + ((42 / 785) * Ydim) - 2, XMin7 + ((302 / 1234) * Xdim) + 2,
                 YMin7 + ((54 / 785) * Ydim) + 2))), 'ASA Classification: 3', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin7 + ((7 / 1234) * Xdim) - 2, YMin7 + ((52 / 785) * Ydim) - 2, XMin7 + ((302 / 1234) * Xdim) + 2,
                 YMin7 + ((63 / 785) * Ydim) + 2))), 'ASA Classification: 4', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin7 + ((7 / 1234) * Xdim) - 2, YMin7 + ((64 / 785) * Ydim) - 2, XMin7 + ((302 / 1234) * Xdim) + 2,
                 YMin7 + ((76 / 785) * Ydim) + 2))), 'ASA Classification: 5', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin7 + ((7 / 1234) * Xdim) - 2, YMin7 + ((83 / 785) * Ydim) - 2, XMin7 + ((302 / 1234) * Xdim) + 2,
                 YMin7 + ((93 / 785) * Ydim) + 2))), 'ASA Classification: E', patientID, flowsheetID, 4)

    # SECTION 8:
    checkbox_function(np.array(
        im.crop((XMin8 + ((130 / 1234) * Xdim) - 2, YMin8 + ((20 / 785) * Ydim) - 2, XMin8 + ((182 / 1234) * Xdim) + 2,
                 YMin8 + ((29 / 785) * Ydim) + 2))), 'Consent Checklist: Anesthesia Consent Complete', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin8 + ((186 / 1234) * Xdim) - 2, YMin8 + ((20 / 785) * Ydim) - 2, XMin8 + ((266 / 1234) * Xdim) + 2,
                 YMin8 + ((29 / 785) * Ydim) + 2))), 'Consent Checklist: Anesthesia Consent Not yet complete',
        patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin8 + ((130 / 1234) * Xdim) - 2, YMin8 + ((32 / 785) * Ydim) - 2, XMin8 + ((182 / 1234) * Xdim) + 2,
                 YMin8 + ((39 / 785) * Ydim) + 2))), 'Consent Checklist: Surgery Consent Complete', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin8 + ((186 / 1234) * Xdim) - 2, YMin8 + ((32 / 785) * Ydim) - 2, XMin8 + ((266 / 1234) * Xdim) + 2,
                 YMin8 + ((39 / 785) * Ydim) + 2))), 'Consent Checklist: Surgery Consent Not yet complete', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin8 + ((130 / 1234) * Xdim) - 2, YMin8 + ((44 / 785) * Ydim) - 2, XMin8 + ((182 / 1234) * Xdim) + 2,
                 YMin8 + ((51 / 785) * Ydim) + 2))), 'Consent Checklist: Blood Transfusion Consent Complete', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin8 + ((186 / 1234) * Xdim) - 2, YMin8 + ((44 / 785) * Ydim) - 2, XMin8 + ((266 / 1234) * Xdim) + 2,
                 YMin8 + ((51 / 785) * Ydim) + 2))), 'Consent Checklist: Blood Transfusion Consent Not yet complete',
        patientID, flowsheetID, 4)

    # SECTION 9:
    checkbox_function(np.array(
        im.crop((XMin9 + ((141 / 1234) * Xdim) - 2, YMin9 + ((5 / 785) * Ydim) - 2, XMin9 + ((190 / 1234) * Xdim) + 2,
                 YMin9 + ((12 / 785) * Ydim) + 2))), 'Planned Anesthesia: General', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((194 / 1234) * Xdim) - 2, YMin9 + ((5 / 785) * Ydim) - 2, XMin9 + ((235 / 1234) * Xdim) + 2,
                 YMin9 + ((12 / 785) * Ydim) + 2))), 'Planned Anesthesia: Spinal', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((240 / 1234) * Xdim) - 2, YMin9 + ((5 / 785) * Ydim) - 2, XMin9 + ((280 / 1234) * Xdim) + 2,
                 YMin9 + ((12 / 785) * Ydim) + 2))), 'Planned Anesthesia: Epidural', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((294 / 1234) * Xdim) - 2, YMin9 + ((5 / 785) * Ydim) - 2, XMin9 + ((330 / 1234) * Xdim) + 2,
                 YMin9 + ((12 / 785) * Ydim) + 2))), 'Planned Anesthesia: Block', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((337 / 1234) * Xdim) - 2, YMin9 + ((5 / 785) * Ydim) - 2, XMin9 + ((380 / 1234) * Xdim) + 2,
                 YMin9 + ((12 / 785) * Ydim) + 2))), 'Planned Anesthesia: Local', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((379 / 1234) * Xdim) - 2, YMin9 + ((5 / 785) * Ydim) - 2, XMin9 + ((433 / 1234) * Xdim) + 2,
                 YMin9 + ((12 / 785) * Ydim) + 2))), 'Planned Anesthesia: Sedation', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((435 / 1234) * Xdim) - 2, YMin9 + ((5 / 785) * Ydim) - 2, XMin9 + ((570 / 1234) * Xdim) + 2,
                 YMin9 + ((12 / 785) * Ydim) + 2))), 'Planned Anesthesia: Other', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((135 / 1234) * Xdim) - 2, YMin9 + ((22 / 785) * Ydim) - 2, XMin9 + ((171 / 1234) * Xdim) + 2,
                 YMin9 + ((31 / 785) * Ydim) + 2))), 'Plan for Procedure Includes: NPO', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((178 / 1234) * Xdim) - 2, YMin9 + ((22 / 785) * Ydim) - 2, XMin9 + ((267 / 1234) * Xdim) + 2,
                 YMin9 + ((31 / 785) * Ydim) + 2))), 'Plan for Procedure Includes: Booking for Blood', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((268 / 1234) * Xdim) - 2, YMin9 + ((22 / 785) * Ydim) - 2, XMin9 + ((360 / 1234) * Xdim) + 2,
                 YMin9 + ((31 / 785) * Ydim) + 2))), 'Plan for Procedure Includes: Special Monitoring', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((360 / 1234) * Xdim) - 2, YMin9 + ((22 / 785) * Ydim) - 2, XMin9 + ((440 / 1234) * Xdim) + 2,
                 YMin9 + ((31 / 785) * Ydim) + 2))), 'Plan for Procedure Includes: New IV Access', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((439 / 1234) * Xdim) - 2, YMin9 + ((22 / 785) * Ydim) - 2, XMin9 + ((572 / 1234) * Xdim) + 2,
                 YMin9 + ((31 / 785) * Ydim) + 2))), 'Plan for Procedure Includes: Other', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((143 / 1234) * Xdim) - 2, YMin9 + ((85 / 785) * Ydim) - 2, XMin9 + ((232 / 1234) * Xdim) + 2,
                 YMin9 + ((92 / 785) * Ydim) + 2))), 'Plan for Post Op Care Includes: Pain Management', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((231 / 1234) * Xdim) - 2, YMin9 + ((85 / 785) * Ydim) - 2, XMin9 + ((258 / 1234) * Xdim) + 2,
                 YMin9 + ((92 / 785) * Ydim) + 2))), 'Plan for Post Op Care Includes: IV', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((257 / 1234) * Xdim) - 2, YMin9 + ((85 / 785) * Ydim) - 2, XMin9 + ((286 / 1234) * Xdim) + 2,
                 YMin9 + ((92 / 785) * Ydim) + 2))), 'Plan for Post Op Care Includes: PO', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((285 / 1234) * Xdim) - 2, YMin9 + ((85 / 785) * Ydim) - 2, XMin9 + ((322 / 1234) * Xdim) + 2,
                 YMin9 + ((92 / 785) * Ydim) + 2))), 'Plan for Post Op Care Includes: Block', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((324 / 1234) * Xdim) - 2, YMin9 + ((85 / 785) * Ydim) - 2, XMin9 + ((374 / 1234) * Xdim) + 2,
                 YMin9 + ((92 / 785) * Ydim) + 2))), 'Plan for Post Op Care Includes: Epidural', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((390 / 1234) * Xdim) - 2, YMin9 + ((85 / 785) * Ydim) - 2, XMin9 + ((479 / 1234) * Xdim) + 2,
                 YMin9 + ((92 / 785) * Ydim) + 2))), 'Plan for Post Op Care Includes: PONV Prevention', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((479 / 1234) * Xdim) - 2, YMin9 + ((85 / 785) * Ydim) - 2, XMin9 + ((566 / 1234) * Xdim) + 2,
                 YMin9 + ((92 / 785) * Ydim) + 2))), 'Plan for Post Op Care Includes: DVT Prophylaxis', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((165 / 1234) * Xdim) - 2, YMin9 + ((96 / 785) * Ydim) - 2, XMin9 + ((236 / 1234) * Xdim) + 2,
                 YMin9 + ((105 / 785) * Ydim) + 2))), 'Post Op Disposition: Day Surgery', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((244 / 1234) * Xdim) - 2, YMin9 + ((96 / 785) * Ydim) - 2, XMin9 + ((322 / 1234) * Xdim) + 2,
                 YMin9 + ((105 / 785) * Ydim) + 2))), 'Post Op Disposition: Surgery Ward', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((329 / 1234) * Xdim) - 2, YMin9 + ((96 / 785) * Ydim) - 2, XMin9 + ((409 / 1234) * Xdim) + 2,
                 YMin9 + ((105 / 785) * Ydim) + 2))), 'Post Op Disposition: ICU Admission', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin9 + ((417 / 1234) * Xdim) - 2, YMin9 + ((96 / 785) * Ydim) - 2, XMin9 + ((561 / 1234) * Xdim) + 2,
                 YMin9 + ((105 / 785) * Ydim) + 2))), 'Post Op Disposition: Other', patientID, flowsheetID, 4)

    # SECTION 10:
    checkbox_function(np.array(im.crop(
        (XMin10 + ((94 / 1234) * Xdim) - 2, YMin10 + ((25 / 785) * Ydim) - 2, XMin10 + ((136 / 1234) * Xdim) + 2,
         YMin10 + ((35 / 785) * Ydim) + 2))), 'Intervention: Elective', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin10 + ((145 / 1234) * Xdim) - 2, YMin10 + ((25 / 785) * Ydim) - 2, XMin10 + ((185 / 1234) * Xdim) + 2,
         YMin10 + ((35 / 785) * Ydim) + 2))), 'Intervention: Urgent', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin10 + ((192 / 1234) * Xdim) - 2, YMin10 + ((25 / 785) * Ydim) - 2, XMin10 + ((252 / 1234) * Xdim) + 2,
         YMin10 + ((35 / 785) * Ydim) + 2))), 'Intervention: Emergent', patientID, flowsheetID, 4)

    # SECTION 11:
    checkbox_function(np.array(
        im.crop((XMin11 + ((80 / 1234) * Xdim) - 2, YMin11 + ((3 / 785) * Ydim) - 2, XMin11 + ((110 / 1234) * Xdim) + 2,
                 YMin11 + ((14 / 785) * Ydim) + 2))), 'Allergies: Yes', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((150 / 1234) * Xdim) - 2, YMin11 + ((3 / 785) * Ydim) - 2, XMin11 + ((210 / 1234) * Xdim) + 2,
         YMin11 + ((14 / 785) * Ydim) + 2))), 'Allergies: No', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((183 / 1234) * Xdim) - 2, YMin11 + ((21 / 785) * Ydim) - 2, XMin11 + ((243 / 1234) * Xdim) + 2,
         YMin11 + ((29 / 785) * Ydim) + 2))), 'Allergies (1): Hives/Itching', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((245 / 1234) * Xdim) - 2, YMin11 + ((21 / 785) * Ydim) - 2, XMin11 + ((315 / 1234) * Xdim) + 2,
         YMin11 + ((29 / 785) * Ydim) + 2))), 'Allergies (1): Mouth Swelling', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((315 / 1234) * Xdim) - 2, YMin11 + ((21 / 785) * Ydim) - 2, XMin11 + ((394 / 1234) * Xdim) + 2,
         YMin11 + ((29 / 785) * Ydim) + 2))), 'Allergies (1): Difficulty Breathing', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((395 / 1234) * Xdim) - 2, YMin11 + ((21 / 785) * Ydim) - 2, XMin11 + ((513 / 1234) * Xdim) + 2,
         YMin11 + ((29 / 785) * Ydim) + 2))), 'Allergies (1): Other', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((516 / 1234) * Xdim) - 2, YMin11 + ((21 / 785) * Ydim) - 2, XMin11 + ((560 / 1234) * Xdim) + 2,
         YMin11 + ((29 / 785) * Ydim) + 2))), 'Allergies (1): Unknown', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((183 / 1234) * Xdim) - 2, YMin11 + ((38 / 785) * Ydim) - 2, XMin11 + ((243 / 1234) * Xdim) + 2,
         YMin11 + ((47 / 785) * Ydim) + 2))), 'Allergies (2): Hives/Itching', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((245 / 1234) * Xdim) - 2, YMin11 + ((38 / 785) * Ydim) - 2, XMin11 + ((315 / 1234) * Xdim) + 2,
         YMin11 + ((47 / 785) * Ydim) + 2))), 'Allergies (2): Mouth Swelling', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((315 / 1234) * Xdim) - 2, YMin11 + ((38 / 785) * Ydim) - 2, XMin11 + ((394 / 1234) * Xdim) + 2,
         YMin11 + ((47 / 785) * Ydim) + 2))), 'Allergies (2): Difficulty Breathing', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((395 / 1234) * Xdim) - 2, YMin11 + ((38 / 785) * Ydim) - 2, XMin11 + ((513 / 1234) * Xdim) + 2,
         YMin11 + ((47 / 785) * Ydim) + 2))), 'Allergies (2): Other', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin11 + ((516 / 1234) * Xdim) - 2, YMin11 + ((38 / 785) * Ydim) - 2, XMin11 + ((560 / 1234) * Xdim) + 2,
         YMin11 + ((47 / 785) * Ydim) + 2))), 'Allergies (2): Unknown', patientID, flowsheetID, 4)

    # SECTION 12:
    checkbox_function(np.array(im.crop(
        (XMin12 + ((148 / 1234) * Xdim) - 2, YMin12 + ((5 / 785) * Ydim) - 2, XMin12 + ((178 / 1234) * Xdim) + 2,
         YMin12 + ((11 / 785) * Ydim) + 2))), 'Past Medical History: Yes', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((218 / 1234) * Xdim) - 2, YMin12 + ((5 / 785) * Ydim) - 2, XMin12 + ((240 / 1234) * Xdim) + 2,
         YMin12 + ((11 / 785) * Ydim) + 2))), 'Past Medical History: No', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((239 / 1234) * Xdim) - 2, YMin12 + ((5 / 785) * Ydim) - 2, XMin12 + ((294 / 1234) * Xdim) + 2,
         YMin12 + ((11 / 785) * Ydim) + 2))), 'Past Medical History: Unknown', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin12 + ((2 / 1234) * Xdim) - 1, YMin12 + ((20 / 785) * Ydim) - 2, XMin12 + ((66 / 1234) * Xdim) + 2,
                 YMin12 + ((28 / 785) * Ydim) + 2))), 'Past Medical History: Smoking', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin12 + ((2 / 1234) * Xdim) - 1, YMin12 + ((31 / 785) * Ydim) - 2, XMin12 + ((66 / 1234) * Xdim) + 2,
                 YMin12 + ((39 / 785) * Ydim) + 2))), 'Past Medical History: Alcohol', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin12 + ((2 / 1234) * Xdim) - 1, YMin12 + ((43 / 785) * Ydim) - 2, XMin12 + ((66 / 1234) * Xdim) + 2,
                 YMin12 + ((51 / 785) * Ydim) + 2))), 'Past Medical History: Toxicomania', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin12 + ((2 / 1234) * Xdim) - 1, YMin12 + ((55 / 785) * Ydim) - 2, XMin12 + ((66 / 1234) * Xdim) + 2,
                 YMin12 + ((63 / 785) * Ydim) + 2))), 'Past Medical History: Asthma', patientID, flowsheetID, 4)

    checkbox_function(np.array(
        im.crop((XMin12 + ((2 / 1234) * Xdim) - 1, YMin12 + ((66 / 785) * Ydim) - 2, XMin12 + ((66 / 1234) * Xdim) + 2,
                 YMin12 + ((74 / 785) * Ydim) + 2))), 'Past Medical History: COPD', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((72 / 1234) * Xdim) - 2, YMin12 + ((20 / 785) * Ydim) - 2, XMin12 + ((146 / 1234) * Xdim) + 2,
         YMin12 + ((28 / 785) * Ydim) + 2))), 'Past Medical History: Tuberculosis', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((72 / 1234) * Xdim) - 2, YMin12 + ((31 / 785) * Ydim) - 2, XMin12 + ((146 / 1234) * Xdim) + 2,
         YMin12 + ((39 / 785) * Ydim) + 2))), 'Past Medical History: Difficult Airway', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((72 / 1234) * Xdim) - 2, YMin12 + ((43 / 785) * Ydim) - 2, XMin12 + ((146 / 1234) * Xdim) + 2,
         YMin12 + ((51 / 785) * Ydim) + 2))), 'Past Medical History: Hypertension', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((72 / 1234) * Xdim) - 2, YMin12 + ((55 / 785) * Ydim) - 2, XMin12 + ((146 / 1234) * Xdim) + 2,
         YMin12 + ((63 / 785) * Ydim) + 2))), 'Past Medical History: MI', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((72 / 1234) * Xdim) - 2, YMin12 + ((66 / 785) * Ydim) - 2, XMin12 + ((146 / 1234) * Xdim) + 2,
         YMin12 + ((74 / 785) * Ydim) + 2))), 'Past Medical History: Angina', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((151 / 1234) * Xdim) - 2, YMin12 + ((20 / 785) * Ydim) - 2, XMin12 + ((252 / 1234) * Xdim) + 2,
         YMin12 + ((28 / 785) * Ydim) + 2))), 'Past Medical History: Arrhythmia', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((151 / 1234) * Xdim) - 2, YMin12 + ((31 / 785) * Ydim) - 2, XMin12 + ((252 / 1234) * Xdim) + 2,
         YMin12 + ((39 / 785) * Ydim) + 2))), 'Past Medical History: Heart Failure', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((151 / 1234) * Xdim) - 2, YMin12 + ((43 / 785) * Ydim) - 2, XMin12 + ((252 / 1234) * Xdim) + 2,
         YMin12 + ((51 / 785) * Ydim) + 2))), 'Past Medical History: Valvulopathy', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((151 / 1234) * Xdim) - 2, YMin12 + ((55 / 785) * Ydim) - 2, XMin12 + ((252 / 1234) * Xdim) + 2,
         YMin12 + ((63 / 785) * Ydim) + 2))), 'Past Medical History: Cardiac Malformation', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((151 / 1234) * Xdim) - 2, YMin12 + ((66 / 785) * Ydim) - 2, XMin12 + ((252 / 1234) * Xdim) + 2,
         YMin12 + ((74 / 785) * Ydim) + 2))), 'Past Medical History: DVT', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((255 / 1234) * Xdim) - 2, YMin12 + ((20 / 785) * Ydim) - 2, XMin12 + ((352 / 1234) * Xdim) + 2,
         YMin12 + ((28 / 785) * Ydim) + 2))), 'Past Medical History: Transfusion', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((255 / 1234) * Xdim) - 2, YMin12 + ((31 / 785) * Ydim) - 2, XMin12 + ((352 / 1234) * Xdim) + 2,
         YMin12 + ((39 / 785) * Ydim) + 2))), 'Past Medical History: Transfusion Reaction', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((255 / 1234) * Xdim) - 2, YMin12 + ((43 / 785) * Ydim) - 2, XMin12 + ((352 / 1234) * Xdim) + 2,
         YMin12 + ((51 / 785) * Ydim) + 2))), 'Past Medical History: Abnormal Breathing', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((255 / 1234) * Xdim) - 2, YMin12 + ((55 / 785) * Ydim) - 2, XMin12 + ((352 / 1234) * Xdim) + 2,
         YMin12 + ((63 / 785) * Ydim) + 2))), 'Past Medical History: Stroke', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((255 / 1234) * Xdim) - 2, YMin12 + ((66 / 785) * Ydim) - 2, XMin12 + ((352 / 1234) * Xdim) + 2,
         YMin12 + ((74 / 785) * Ydim) + 2))), 'Past Medical History: Epilepsy', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((352 / 1234) * Xdim) - 2, YMin12 + ((20 / 785) * Ydim) - 2, XMin12 + ((440 / 1234) * Xdim) + 2,
         YMin12 + ((28 / 785) * Ydim) + 2))), 'Past Medical History: Diabetes', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((352 / 1234) * Xdim) - 2, YMin12 + ((31 / 785) * Ydim) - 2, XMin12 + ((440 / 1234) * Xdim) + 2,
         YMin12 + ((39 / 785) * Ydim) + 2))), 'Past Medical History: Thyroid Disease', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((352 / 1234) * Xdim) - 2, YMin12 + ((43 / 785) * Ydim) - 2, XMin12 + ((440 / 1234) * Xdim) + 2,
         YMin12 + ((51 / 785) * Ydim) + 2))), 'Past Medical History: Renal Failure', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((352 / 1234) * Xdim) - 2, YMin12 + ((55 / 785) * Ydim) - 2, XMin12 + ((440 / 1234) * Xdim) + 2,
         YMin12 + ((63 / 785) * Ydim) + 2))), 'Past Medical History: Gastric Ulcer', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((352 / 1234) * Xdim) - 2, YMin12 + ((66 / 785) * Ydim) - 2, XMin12 + ((440 / 1234) * Xdim) + 2,
         YMin12 + ((74 / 785) * Ydim) + 2))), 'Past Medical History: Esophageal Reflux', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((444 / 1234) * Xdim) - 2, YMin12 + ((20 / 785) * Ydim) - 2, XMin12 + ((538 / 1234) * Xdim) + 2,
         YMin12 + ((28 / 785) * Ydim) + 2))), 'Past Medical History: Cirrhosis', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((444 / 1234) * Xdim) - 2, YMin12 + ((31 / 785) * Ydim) - 2, XMin12 + ((538 / 1234) * Xdim) + 2,
         YMin12 + ((39 / 785) * Ydim) + 2))), 'Past Medical History: Hepatitis', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((444 / 1234) * Xdim) - 2, YMin12 + ((43 / 785) * Ydim) - 2, XMin12 + ((538 / 1234) * Xdim) + 2,
         YMin12 + ((51 / 785) * Ydim) + 2))), 'Past Medical History: HIV', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((444 / 1234) * Xdim) - 2, YMin12 + ((55 / 785) * Ydim) - 2, XMin12 + ((538 / 1234) * Xdim) + 2,
         YMin12 + ((63 / 785) * Ydim) + 2))), 'Past Medical History: Prosthetic Teeth', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin12 + ((444 / 1234) * Xdim) - 2, YMin12 + ((66 / 785) * Ydim) - 2, XMin12 + ((538 / 1234) * Xdim) + 2,
         YMin12 + ((74 / 785) * Ydim) + 2))), 'Past Medical History: Other', patientID, flowsheetID, 4)

    # SECTION 13:
    checkbox_function(np.array(im.crop(
        (XMin13 + ((213 / 1234) * Xdim) - 2, YMin13 + ((5 / 785) * Ydim) - 2, XMin13 + ((241 / 1234) * Xdim) + 2,
         YMin13 + ((13 / 785) * Ydim) + 2))), 'Surgical & Anesthetic History: Yes', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin13 + ((295 / 1234) * Xdim) - 2, YMin13 + ((5 / 785) * Ydim) - 2, XMin13 + ((319 / 1234) * Xdim) + 2,
         YMin13 + ((13 / 785) * Ydim) + 2))), 'Surgical & Anesthetic History: No', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin13 + ((334 / 1234) * Xdim) - 2, YMin13 + ((5 / 785) * Ydim) - 2, XMin13 + ((388 / 1234) * Xdim) + 2,
         YMin13 + ((13 / 785) * Ydim) + 2))), 'Surgical & Anesthetic History: Unknown', patientID, flowsheetID, 4)

    # SECTION 14:
    checkbox_function(np.array(im.crop(
        (XMin14 + ((150 / 1234) * Xdim) - 2, YMin14 + ((6 / 785) * Ydim) - 2, XMin14 + ((186 / 1234) * Xdim) + 2,
         YMin14 + ((15 / 785) * Ydim) + 2))), 'Current Medications: Yes', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin14 + ((242 / 1234) * Xdim) - 2, YMin14 + ((6 / 785) * Ydim) - 2, XMin14 + ((273 / 1234) * Xdim) + 2,
         YMin14 + ((15 / 785) * Ydim) + 2))), 'Current Medications: No', patientID, flowsheetID, 4)

    checkbox_function(np.array(im.crop(
        (XMin14 + ((286 / 1234) * Xdim) - 2, YMin14 + ((6 / 785) * Ydim) - 2, XMin14 + ((344 / 1234) * Xdim) + 2,
         YMin14 + ((15 / 785) * Ydim) + 2))), 'Current Medications: Unknown', patientID, flowsheetID, 4)

def checkSide(imageA):
    refFilename = "C:/Users/maryblankemeier/Desktop/IntegrationPackage/IntraoperativeForm.jpg"
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    print(imageA)
    refFileWidth = imReference.shape[1]

  
    imPatientOrg = cv2.imread(imageA, cv2.IMREAD_COLOR) #image passing in
    #print(imPatientOrg)
    imPatient = image_resize(imPatientOrg, width=refFileWidth)
    img, h = alignImages(imPatient, imReference, MAX_FEATURES=500, GOOD_MATCH_PERCENT=0.1)
    # img = imPatient
    im = Image.fromarray(img)

    standImg = cv2.imread(refFilename)
    standIm = Image.open(refFilename)
    edges = cv2.Canny(standImg, 50, 110)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=300, lines=np.array([]), minLineLength=1,
                            maxLineGap=50)

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

    intraopRecordArea = im.crop((0, 0, 250, minY))
    ratio = checkPixelRatio(intraopRecordArea)
    if ratio > 0.2: #swap
        #print('by')
        return True
    else:
        #print('hi')
        return False #switched that 

def checkPixelRatio(image):
    pixels = ma.getdata(image)  # get the pixels as a flattened sequence
    black_thresh = 125
    nblack = 0
    num_pix = 0
    for pixel in pixels:
        for p in pixel:
            for each in p:
                if each < black_thresh:
                    nblack += 1
                num_pix += 1
    ratio = nblack / float(num_pix)
    return ratio

def integrated_function(flowsheet, patientID, timestamp): #main method

    shutil.move('IntegrationPackage/NewFlowsheets/' + flowsheet, 'IntegrationPackage')
    #print("done")

    flowsheetID = patientID + ' - ' + timestamp
    print(flowsheetID)

    # checking if the patient id exists, if it exists then increment
    exists_query = '''
        select exists (
            select 1
            from patient_identity
            where PATIENT_ID = %s
        )'''

    cur.execute(exists_query, (patientID,))
    if cur.fetchone()[0] == False:
        # Inserting into Patient Identity
        postgres_insert_query_0 = """ INSERT INTO patient_identity (PATIENT_ID, ENTRY_COUNT) VALUES (%s,%s)"""
        record_to_insert_0 = (patientID, 1)
        cur.execute(postgres_insert_query_0, record_to_insert_0)

        ##Committing an insert###
        con.commit()
        count = cur.rowcount
        print(count, "Record inserted successfully into Patient Identity Table")

    else:  # we want to add to the count here
        update_count = """ UPDATE patient_identity
                SET ENTRY_COUNT = ENTRY_COUNT+1
                WHERE PATIENT_ID = %s"""

        cur.execute(update_count, (patientID,))

    # Inserting into Flowsheet
    postgres_insert_query_0 = """ INSERT INTO flowsheet_id (FLOWSHEET_ID, PATIENT_ID, TIMESTAMP) VALUES (%s,%s,%s)"""
    record_to_insert_0 = (flowsheetID, patientID, timestamp)
    cur.execute(postgres_insert_query_0, record_to_insert_0)

    ##Committing an insert###
    con.commit()
    count = cur.rowcount
    print(count, "Record inserted successfully into Flowsheet Identity Table")
    
    checkSide(flowsheet)
    
    crop_intraop(flowsheet, patientID, flowsheetID)
    #print(checkSide(flowsheet))
    #if checkSide(flowsheet):
    #    print('worked')
    #    crop_intraop(flowsheet, patientID, flowsheetID)
        
   # else:
     #  print('failed')
       #crop_anesthesia(flowsheet, patientID, flowsheetID)

    shutil.move('IntegrationPackage/' + flowsheet, 'IntegrationPackage/ProcessedFlowsheets')

def decrypt(folder, filename):
    chunksize =64 * 1024
    outputFile = filename[11:]


    private_key = RSA.import_key(open("C:/Users/maryblankemeier/Desktop/IntegrationPackage/private.pem").read())
    cipher_rsa = PKCS1_OAEP.new(private_key)



    with open(os.path.join(folder, filename), "rb") as infile:
        filesize = int(infile.read(16))

        enc_session_key = infile.read(256)
        #with open('C:/Users/lrm3k/Box Sync/key', "rb") as keyfile:
        #    enc_session_key = keyfile.read()
#        print(filesize)
#        print(len(enc_session_key))
#        print(enc_session_key)

        session_key = cipher_rsa.decrypt(enc_session_key)
#        print(session_key)

        IV = infile.read(16)



        decryptor = AES.new(session_key, AES.MODE_EAX, IV)

        with open(os.path.join(folder, outputFile), "wb") as outfile:
            while True:
                chunk = infile.read(chunksize)

                if len(chunk) == 0:
                    break

                outfile.write(decryptor.decrypt(chunk))
                outfile.truncate(filesize)

    return outputFile

def file_extraction():
    i=0
    org_email="@gmail.com"
    from_email="syscapstone"+org_email
    from_pwd=env_pw
    smtp_server="imap.gmail.com"
    smtp_port=993

    mail=imaplib.IMAP4_SSL(smtp_server)
    mail.login(from_email,from_pwd)

    latest_email_uid = ''
    mail.select('inbox')
    type, data = mail.search(None, '(UNSEEN)')
    ids = data[0]
    id_list = ids.split()
    
    if len(id_list) == 0:
        return

    while True:
#        print(i)
        #type, data = mail.uid('search', None, "ALL")
#        print(id_list)
#        print(latest_email_uid)
        if id_list[-1] == latest_email_uid:
            return
            #time.sleep(600)
        else:
            latest_email_uid = id_list[i]
            type, data = mail.fetch(latest_email_uid, '(RFC822)' )
            #type, data = mail.uid('fetch',str(latest_email_uid),"(RFC822)")
            raw_email = data[0][1]
            # converts byte literal to string removing b''
            raw_email_string = raw_email.decode('utf-8')
            email_message = email.message_from_string(raw_email_string)
            # downloading attachments
            timestamp = email_message['Date']
            print(email_message['Date'])
            subject = str(email.header.make_header(email.header.decode_header(email_message['Subject'])))
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body = str(part.get_payload(decode=True))
#                    print(body.strip())
                if part.get_content_maintype() == 'multipart':
                    continue
                if part.get('Content-Disposition') is None:
                    continue
                fileName = part.get_filename()
                #print(fileName)
                if bool(fileName):
                    filePath = os.path.join('IntegrationPackage/NewFlowsheets', fileName)
                    if not os.path.isfile(filePath) :
                        fp = open(filePath, 'wb')
                        fp.write(part.get_payload(decode=True))
                        fp.close()
                    decoded = decrypt('IntegrationPackage/NewFlowsheets', fileName)
                    #print(os.path.exists(os.path.join('C:/Users/maryblankemeier/Desktop/IntegrationPackage/NewFlowsheets', '026.jpg')))

                    integrated_function(decoded, subject, timestamp)

            i=i+1

file_extraction()

