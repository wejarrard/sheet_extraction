"""
Returns a pandas dataframe from the results of the drug extraction
"""
from __future__ import print_function
from PIL import Image
import numpy as np
import cv2
import os
from itertools import *
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
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

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, _ = im2.shape
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


def crop_intraop(image_path, refFilename, imReference, refFileWidth):
    imFilename = image_path
    imPatientOrg = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    patient = os.path.basename(imFilename)[:-4]
    imPatient = image_resize(imPatientOrg, width=refFileWidth)

    # Registered image will be restored in img.
    # img = aligned image (cv2)
    # im = aligned image (PIL)
    # The estimated homography will be stored in h.
    img, h = alignImages(
        imPatient, imReference, MAX_FEATURES=500, GOOD_MATCH_PERCENT=0.16
    )
    im = Image.fromarray(img)

    # determine chart pixel area (on standard form)
    standImg = cv2.imread(refFilename)
    edges = cv2.Canny(standImg, 50, 110)
    lines = cv2.HoughLinesP(
        image=edges,
        rho=1,
        theta=np.pi / 180,
        threshold=300,
        lines=np.array([]),
        minLineLength=1,
        maxLineGap=50,
    )

    # determine chart edges -> based on location in IntraoperativeRecord.jpg (or standard form)
    minX = lines[0][0][0]
    minY = lines[0][0][1]
    maxX = lines[0][0][0]
    maxY = lines[0][0][1]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 < minX:
            minX = x1
        if x2 > maxX:
            maxX = x2
        if y1 < minY:
            minY = y1
        if y2 > maxY:
            maxY = y2

    Xdim = maxX - minX
    Ydim = maxY - minY

    edgeImg = img
    cv2.line(edgeImg, (minX, minY), (minX, maxY), (0, 0, 255), 2)
    cv2.line(edgeImg, (maxX, minY), (maxX, maxY), (0, 0, 255), 2)
    cv2.line(edgeImg, (minX, minY), (maxX, minY), (0, 0, 255), 2)
    cv2.line(edgeImg, (minX, maxY), (maxX, maxY), (0, 0, 255), 2)

    # Show Flowsheet Sections
    # SECTION 1 - Ch
    XMin1 = minX + int((20 / 1190) * Xdim)
    XMax1 = minX + int((116 / 1190) * Xdim)
    YMin1 = minY + int((20 / 756) * Ydim)
    YMax1 = minY + int((310 / 756) * Ydim)

    checkbox_crop = np.array(im.crop((XMin1 - 20, YMin1 - 60, XMax1 + 70, YMax1 - 100)))
    return checkbox_crop


def find_IVDrugs(checkbox_crop, IV_template):

    IV_checkbox_w, IV_checkbox_h = IV_template.shape[::-1]

    # Crop the image for the correct column
    checkbox_crop2 = checkbox_crop
    im = Image.fromarray(checkbox_crop)
    checkbox_crop = np.array(im.crop((0, 0, 200, 120)))

    img_gray = cv2.cvtColor(checkbox_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    img_gray = clahe.apply(img_gray)

    # Perform match operations.
    res_1 = cv2.matchTemplate(img_gray, IV_template, cv2.TM_CCOEFF_NORMED)

    # Specify a threshold
    threshold_1 = 0.4

    # Store the coordinates of matched area in a numpy array
    loc_1 = np.where(res_1 >= threshold_1)

    # print(np.size(loc))
    detect = chain(zip(*loc_1[::-1]))
    data = sorted(detect)

    count = 0
    mask = np.zeros(img_gray.shape[:2], np.uint8)

    # Draw a rectangle around the matched region.
    for pt in data:  # zip(*loc_1[::-1])

        if (
            mask[pt[1] + round(IV_checkbox_h / 2), pt[0] + round(IV_checkbox_w / 2)]
            != 255
        ):
            mask[pt[1] : pt[1] + IV_checkbox_h, pt[0] : pt[0] + IV_checkbox_w] = 255
            count += 1
            cv2.rectangle(
                img_gray,
                pt,
                (pt[0] + IV_checkbox_w, pt[1] + IV_checkbox_h),
                (0, 255, 255),
                2,
            )

    # If there is a match, return the best fit
    if count >= 1:

        img_gray = cv2.cvtColor(checkbox_crop, cv2.COLOR_BGR2GRAY)
        methods = ["cv2.TM_CCOEFF_NORMED"]

        for meth in methods:
            method = eval(meth)
            # Apply template Matching
            res = cv2.matchTemplate(img_gray, IV_template, method)
            # Return the points that best fit the template match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                point = min_loc
            else:
                point = max_loc
            bottom_right = (point[0] + IV_checkbox_w, point[1] + IV_checkbox_h)
            # cv2.rectangle(checkbox_crop,top_left, bottom_right, 255, 2)
            # Using template matching to crop the checkbox

            cropped_box_1 = checkbox_crop2[
                point[1] + 20 : point[1] + 44,
                point[0] - 60 : point[0] + IV_checkbox_w + 10,
            ]
            cropped_box_2 = checkbox_crop2[
                point[1] + 38 : point[1] + 58,
                point[0] - 60 : point[0] + IV_checkbox_w + 10,
            ]
            cropped_box_3 = checkbox_crop2[
                point[1] + 56 : point[1] + 76,
                point[0] - 60 : point[0] + IV_checkbox_w + 10,
            ]
            cropped_box_4 = checkbox_crop2[
                point[1] + 71 : point[1] + 91,
                point[0] - 60 : point[0] + IV_checkbox_w + 10,
            ]
            cropped_box_5 = checkbox_crop2[
                point[1] + 87 : point[1] + 107,
                point[0] - 60 : point[0] + IV_checkbox_w + 10,
            ]
            cropped_box_6 = checkbox_crop2[
                point[1] + 102 : point[1] + 122,
                point[0] - 60 : point[0] + IV_checkbox_w + 10,
            ]
            cropped_box_7 = checkbox_crop2[
                point[1] + 119 : point[1] + 139,
                point[0] - 60 : point[0] + IV_checkbox_w + 10,
            ]
            cropped_box_8 = checkbox_crop2[
                point[1] + 134 : point[1] + 154,
                point[0] - 60 : point[0] + IV_checkbox_w + 10,
            ]

            return (
                cropped_box_1,
                cropped_box_2,
                cropped_box_3,
                cropped_box_4,
                cropped_box_5,
                cropped_box_6,
                cropped_box_7,
                cropped_box_8,
            )

    else:
        print("None found")
        return 0, 0, 0, 0, 0, 0, 0, 0


def drug_extration(imagePath):

    refFilename = "../templates/IntraoperativeForm.JPG"
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    refFileWidth = imReference.shape[1]

    # Load Templates

    IV_template = cv2.imread("../templates/Template - Time.JPG", 0)

    priors = pd.read_csv("../templates/Line_priors.csv")
    first_threshold = 0.65
    second_threshold = 0.45
    prior_weight = 0.75

    # Set list of images
    list_of_image_paths = glob.glob(imagePath)

    text_cnn = load_model("../models/Full_VGG16.h5")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    class_names = [
        "Atropine",
        "Blanks",
        "Cefazolin",
        "Cefotaxime",
        "Ceftriaxone",
        "Dexamethasone",
        "Diclofenac",
        "Ephedrine",
        "Fentanyl",
        "Hydrocortisone",
        "Ketamine",
        "Lidocaine",
        "Marcaine",
        "Metronidazole",
        "Midazolam",
        "Morphine",
        "Neostigmine",
        "Paracetamol",
        "Propofol",
        "Suxamethonium",
        "Thiopental",
        "Vecuronium",
    ]

    records = pd.DataFrame(
        columns=[
            "Flowsheet_ID",
            "Line1",
            "Line2",
            "Line3",
            "Line4",
            "Line5",
            "Line6",
            "Line7",
            "Line8",
        ]
    )

    for i in list_of_image_paths:
        print(i)
        checkbox_crop = crop_intraop(i, refFilename, imReference, refFileWidth)
        file_name = os.path.basename(i)[:-4]

        (
            cropped_box_1,
            cropped_box_2,
            cropped_box_3,
            cropped_box_4,
            cropped_box_5,
            cropped_box_6,
            cropped_box_7,
            cropped_box_8,
        ) = find_IVDrugs(checkbox_crop, IV_template)
        plt.show()

        # Create dataframe to hold predictions for each flowsheet, should be reset for each flowsheet
        flowsheet = pd.DataFrame(columns=class_names)

        # Create list of the crops
        boxes = [
            cropped_box_1,
            cropped_box_2,
            cropped_box_3,
            cropped_box_4,
            cropped_box_5,
            cropped_box_6,
            cropped_box_7,
            cropped_box_8,
        ]

        # Generate predictions for each line on the flowsheet
        for box in boxes:
            if np.mean(box) == 0 or box.shape[1] == 0:
                flowsheet.loc[len(flowsheet.index)] = np.zeros(22)
            else:
                X = np.zeros(
                    [1, box.shape[0], box.shape[1], box.shape[2]], dtype=np.uint8
                )
                X[0, : box.shape[0], : box.shape[1], :] = box
                X = tf.image.resize(X, [32, 128])
                test_generator = test_datagen.flow(X, batch_size=1)
                pred = text_cnn.predict(test_generator)
                flowsheet.loc[len(flowsheet.index)] = pred[0]

        # Calculate the max prediction for each line on the flowsheet
        flowsheet["max_prediction"] = flowsheet.iloc[:, :22].max(axis=1)
        flowsheet["max_prediction_class"] = np.argmax(
            np.array(flowsheet.iloc[:, :22]), axis=1
        )
        flowsheet["max_prediction_name"] = flowsheet.max_prediction_class.apply(
            lambda x: class_names[x]
        )

        # Calculate the combined probability with predictions and priors
        flowsheet2 = (
            priors.iloc[:, 1:23] * prior_weight
            + (1 - prior_weight) * flowsheet.iloc[:, :22]
        )
        flowsheet2["max_prediction"] = flowsheet2.iloc[:, :22].max(axis=1)
        flowsheet2["max_prediction_class"] = np.argmax(
            np.array(flowsheet2.iloc[:, :22]), axis=1
        )
        flowsheet2["max_prediction_name"] = flowsheet2.max_prediction_class.apply(
            lambda x: class_names[x]
        )

        # Add the file name that will be used for the output row
        row = [file_name]

        # Use thresholds to determine the most likely drug in each line on the flowsheet and save to dataframe
        for i in range(8):
            if flowsheet.iloc[i, 22] >= first_threshold:
                row += [flowsheet.iloc[i, 24]]
            elif flowsheet2.iloc[i, 22] >= second_threshold:
                row += [flowsheet2.iloc[0, 24]]
            else:
                row += ["Unsure"]
        records.loc[len(records.index)] = row

    return records


if __name__ == "__main__":
    drug_extration("../images/128.jpg")
