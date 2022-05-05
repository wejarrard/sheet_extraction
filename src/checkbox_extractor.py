"""checkbox_extractor.py

This script looks for all photographs/scans of perioperative flowsheets in
the folder "new_sheets", processes them, and creates a csv file called
'Predicted_Valuesv2.csv' containing a row for each flowsheet in the
NewSheets folder. Each column corresponds to a checkbox on the flowsheet
and has either a True/False or None value.

You will need the checkbox model 'ResNet-imp-.9532.h5' and all the template
images to run this file.
"""


from itertools import chain
import glob
import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


refFilename = "../templates/IntraoperativeForm.JPG"
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
refFileWidth = imReference.shape[1]

templates = {}
for i in glob.glob("../templates/checkbox_templates/*.jpg"):
    templates[os.path.basename(i)] = cv2.imread(i, 0)

column_coordinates = {
    "PATIENT_SAFETY": (0, 10, 150, 80),
    "MASK_VENTILATION": (80, 10, 280, 75),
    "AIRWAY": (200, 10, 440, 100),
    "AIRWAY_PLACEMENT_AID": (360, 10, 700, 100),
    "LRA": (570, 10, 900, 100),
    "TUBES_AND_LINES": (730, 10, 1050, 100),
    "MONITORING_DETAILS_LEFT_COLUMN": (800, 10, 1150, 100),
    "MONITORING_DETAILS_RIGHT_COLUMN": (900, 10, 1300, 100),
    "PATIENT_POSITION_LEFT_COLUMN": (1000, 0, 1250, 120),
    "PATIENT_POSITION_RIGHT_COLUMN": (1080, 0, 1200, 120),
}
thresholds = {
    "Template - Warming3.JPG": 0.2,
    "Template - TED and Safety.JPG": 0.35,
    "Template - TED3.JPG": 0.45,
    "Template - Safety.JPG": 0.4,
    "Template - Ventiliation.JPG": 0.25,
    "Template - Easy Ventiliation.JPG": 0.25,
    "Template - Oral Airway.JPG": 0.25,
    "Template - Difficult Ventiliation2.JPG": 0.45,
    "Template - Airway.JPG": 0.25,
    "Template - FM.JPG": 0.25,
    "Template - LMA.JPG": 0.32,
    "Template - ETT.JPG": 0.45,
    "Template - Trach.JPG": 0.45,
    "Template - APA.JPG": 0.25,
    "Template - APA Used.JPG": 0.45,
    "Template - Fibroscope.JPG": 0.32,
    "Template - Bronchoscope.JPG": 0.45,
    "Template - OtherView.JPG": 0.45,
    "Template - LRA.JPG": 0.25,
    "Template - TnL.JPG": 0.3,
    "Template - IV.JPG": 0.32,
    "Template - Tubes.JPG": 0.47,
    "Template - MD1 wide.JPG": 0.4,
    "Template - MD1.3.JPG": 0.4,
    "Template - MD1.2 narrow.JPG": 0.55,
    "Template - MD2.JPG": 0.35,
    "Template - MD2.3.JPG": 0.45,
    "Template - MD2.2.JPG": 0.6,
    "Template - PP1 Narrow.JPG": 0.35,
    "Template - PP2 narrow(remake).JPG": 0.43,
}


class checkbox_extractor:
    def __init__(self, flowsheets_folder):
        self.flowsheets_folder = flowsheets_folder
        self.predicted_checkboxes = None

    def extract_checkboxes(self):
        # Set list of images
        list_of_image_paths = glob.glob(self.flowsheets_folder + "/*.JPG")

        checkbox_cnn = tf.keras.models.load_model(
            "../models/ResNet-imp-.9532.h5", compile=False
        )
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255
        )

        columns = [
            "Flowsheet_ID",
            "Eye_Protection",
            "Warming",
            "TED_Stockings",
            "Safety_Checklist",
            "Easy_Ventilation",
            "Vent_Adjunct",
            "Dif_Ventilation",
            "Natural_Airway",
            "LMA",
            "ETT",
            "Trach",
            "APA_Used",
            "Fibroscope",
            "Bronchoscope",
            "APA_Other",
            "LRA_Used",
            "Peripheral_IV",
            "Central_IV",
            "Urinary_Catheter",
            "Gastric_Tube",
            "ECG",
            "NIBP",
            "SpO2",
            "EtCO2",
            "Stethoscope",
            "Temperature",
            "NMT",
            "Urine_Output",
            "Arterial_BP",
            "MD_Other",
            "Supine",
            "Prone",
            "Lithotomy",
            "Sitting",
            "Trendelenburg",
            "Fowler",
            "Lateral",
        ]
        records = pd.DataFrame(columns=columns)

        for i in list_of_image_paths:
            print(i)
            checkbox_crop = self.crop_intraop(i)
            file_name = os.path.basename(i)[:-4]

            box1, box2 = self.find_eye_protection_and_warming(checkbox_crop)
            box3, box4 = self.find_ted_and_safety(checkbox_crop)
            box5, box6, box7 = self.find_ventilation(checkbox_crop)
            box8, box9, box10, box11 = self.find_airway(checkbox_crop)
            box12, box13, box14, box15 = self.find_apa(checkbox_crop)
            box16 = self.find_lra(checkbox_crop)
            box17, box18, box19, box20 = self.find_tubes_and_lines(checkbox_crop)
            box21, box22, box23, box24, box25 = self.find_md1(checkbox_crop)
            box26, box27, box28, box29, box30 = self.find_md2(checkbox_crop)
            box31, box32, box33, box34 = self.find_pp1(checkbox_crop)
            box35, box36, box37 = self.find_pp2(checkbox_crop)

            boxes = [
                box1,
                box2,
                box3,
                box4,
                box5,
                box6,
                box7,
                box8,
                box9,
                box10,
                box11,
                box12,
                box13,
                box14,
                box15,
                box16,
                box17,
                box18,
                box19,
                box20,
                box21,
                box22,
                box23,
                box24,
                box25,
                box26,
                box27,
                box28,
                box29,
                box30,
                box31,
                box32,
                box33,
                box34,
                box35,
                box36,
                box37,
            ]

            row = [file_name]
            for box in boxes:
                if np.mean(box) == 0 or box.size == 0:
                    row += [None]
                else:
                    temp = np.zeros(
                        [1, box.shape[0], box.shape[1], box.shape[2]], dtype=np.uint8
                    )
                    temp[0, : box.shape[0], : box.shape[1], :] = box
                    temp = tf.image.resize(temp, [64, 256])
                    test_generator = test_datagen.flow(temp, batch_size=1)
                    pred = checkbox_cnn.predict(test_generator)
                    checked = pred < 0.5
                    row += [checked[0][0]]

            records.loc[len(records.index)] = row
        self.predicted_checkboxes = records
        return records

    def align_images(self, im1, im2, max_features, good_match_percent):
        """Aligns a new flowsheet to the reference flowsheet."""
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        )
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        num_good_matches = int(len(matches) * good_match_percent)
        matches = matches[:num_good_matches]

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
        im_1_reg = cv2.warpPerspective(im1, h, (width, height))

        return im_1_reg, h

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """Resizes an image to the specified dimensions.

        Keyword arguments:
        image  -- an image processed by the cv2.imread() function
        width  -- the desired width of the output image (default None)
        height -- the desired height of the output image (default None)
        inter  -- the
        """
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (old_height, old_width) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            ratio = height / float(old_height)
            dim = (int(old_height * ratio), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            ratio = width / float(old_width)
            dim = (width, int(old_height * ratio))

        # return the resized image
        return cv2.resize(image, dim, interpolation=inter)

    def crop_intraop(self, image_path):
        im_filename = image_path
        im_patient_org = cv2.imread(im_filename, cv2.IMREAD_COLOR)
        im_patient = self.image_resize(im_patient_org, width=refFileWidth)

        # Registered image will be restored in img.
        # img = aligned image (cv2)
        # image = aligned image (PIL)
        # The estimated homography will be stored in h.
        img, _ = self.align_images(
            im_patient, imReference, max_features=500, good_match_percent=0.1
        )
        image = Image.fromarray(img)

        # determine chart pixel area (on standard form)
        stand_img = cv2.imread(refFilename)
        edges = cv2.Canny(stand_img, 50, 110)
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
        min_x = lines[0][0][0]
        min_y = lines[0][0][1]
        max_x = lines[0][0][0]
        max_y = lines[0][0][1]

        for line in lines:
            x_1, y_1, x_2, y_2 = line[0]
            if x_1 < min_x:
                min_x = x_1
            if x_2 > max_x:
                max_x = x_2
            if y_1 < min_y:
                min_y = y_1
            if y_2 > max_y:
                max_y = y_2

        x_dim = max_x - min_x
        y_dim = max_y - min_y

        cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 0, 255), 2)
        cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 0, 255), 2)
        cv2.line(img, (min_x, max_y), (max_x, max_y), (0, 0, 255), 2)

        # SECTION 6 - all checkboxes
        x_min_6 = min_x + int((0 / 1190) * x_dim)
        x_max_6 = min_x + int((1190 / 1190) * x_dim)
        y_min_6 = min_y + int((587 / 756) * y_dim)
        y_max_6 = min_y + int((696 / 756) * y_dim)

        checkbox_crop = np.array(image.crop((x_min_6, y_min_6, x_max_6, y_max_6 + 15)))
        return checkbox_crop

    def match_is_found(self, checkbox_crop, checkbox_name):
        """"""
        img_gray = cv2.cvtColor(checkbox_crop, cv2.COLOR_BGR2GRAY)

        # Perform match operations. What does this variable name mean???
        res_1 = cv2.matchTemplate(
            img_gray, templates[checkbox_name], cv2.TM_CCOEFF_NORMED
        )

        # Specify a threshold
        threshold = thresholds[checkbox_name]

        template = templates[checkbox_name]
        template_width, template_height = template.shape[::-1]

        # Store the coordinates of matched area in a numpy array
        loc_1 = np.where(res_1 >= threshold)

        data = sorted(chain(zip(*loc_1[::-1])))
        mask = np.zeros(img_gray.shape[:2], np.uint8)

        for point in data:
            if (
                mask[
                    point[1] + round(template_height / 2),
                    point[0] + round(template_width / 2),
                ]
                != 255
            ):
                return True

        return False

    def compute_point(self, checkbox_crop, template_name):
        """Computes the 'point' parameter used by the checkbox methods."""
        img_gray = cv2.cvtColor(checkbox_crop, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(
            img_gray, templates[template_name], cv2.TM_CCOEFF_NORMED
        )
        return cv2.minMaxLoc(res)[3]

    def get_template_height_and_width(self, template_name):
        """Gets the height and width of the template file 'filename'"""
        return templates[template_name].shape[::-1]

    # Patient Safety Column
    def find_eye_protection_and_warming(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(image.crop(column_coordinates["PATIENT_SAFETY"]))
        template_name = "Template - Warming3.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_ep = checkbox_crop[
                point[1] : point[1] + checkbox_height // 2 + 3,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_w = checkbox_crop[
                point[1] + checkbox_height // 2 : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return [cropped_box_ep, cropped_box_w]

        return [0, 0]

    def find_ted_and_safety(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(image.crop(column_coordinates["PATIENT_SAFETY"]))
        template_name = "Template - TED and Safety.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )

            cropped_box_ted = checkbox_crop[
                point[1] : point[1] + checkbox_height // 2 + 3,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_sc = checkbox_crop[
                point[1] + checkbox_height // 2 : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_ted, cropped_box_sc

        print("Step 2 Only")
        return [
            self.find_ted_stockings(checkbox_crop),
            self.find_safety_checklist(checkbox_crop),
        ]

    def find_ted_stockings(self, checkbox_crop):
        # If there is a match, return the best fit
        template_name = "Template - TED3.JPG"
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            cropped_box_ted = checkbox_crop[
                point[1] + checkbox_height // 2 : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_ted

        return 0

    def find_safety_checklist(self, checkbox_crop):
        # If there is a match, return the best fit
        template_name = "Template - Safety.JPG"
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            cropped_box_sc = checkbox_crop[
                point[1] : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_sc

        return 0

    def find_ventilation(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(image.crop(column_coordinates["MASK_VENTILATION"]))
        template_name = "Template - Ventiliation.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, "Template - Ventiliation.JPG"):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            cropped_box_ev = checkbox_crop[
                point[1] : point[1] + checkbox_height // 3 + 2,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_oa = checkbox_crop[
                point[1] + checkbox_height // 3 : point[1] + 2 * checkbox_height // 3,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_dv = checkbox_crop[
                point[1] + checkbox_height // 2 + 2 : point[1] + checkbox_height - 2,
                point[0] : point[0] + checkbox_width,
            ]

            return [cropped_box_ev, cropped_box_oa, cropped_box_dv]

        print("STEP 2 ONLY")
        return [
            self.find_easy_ventilation(checkbox_crop),
            self.find_oa_ventilation(checkbox_crop),
            self.find_difficult_ventilation(checkbox_crop),
        ]

    def find_easy_ventilation(self, checkbox_crop):
        template_name = "Template - Easy Ventiliation.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_ev = checkbox_crop[
                point[1] : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_ev

        return 0

    def find_oa_ventilation(self, checkbox_crop):
        template_name = "Template - Oral Airway.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_oa = checkbox_crop[
                point[1] : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_oa

        return 0

    def find_difficult_ventilation(self, checkbox_crop):
        template_name = "Template - Difficult Ventiliation2.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, "Template - Difficult Ventiliation2.JPG"):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_dv = checkbox_crop[
                point[1] : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_dv

        return 0

    # Airway Column
    def find_airway(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(image.crop(column_coordinates["AIRWAY"]))
        template_name = "Template - Airway.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_fm = checkbox_crop[
                point[1] : point[1] + checkbox_height // 4 + 6,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_lma = checkbox_crop[
                point[1]
                + checkbox_height // 4
                - 3 : point[1]
                + checkbox_height // 2
                + 3,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_ett = checkbox_crop[
                point[1]
                + checkbox_height // 2
                - 3 : point[1]
                + 3 * checkbox_height // 4
                + 3,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_trach = checkbox_crop[
                point[1] + 3 * checkbox_height // 4 - 6 : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return [cropped_box_fm, cropped_box_lma, cropped_box_ett, cropped_box_trach]

        print("STEP 2 ONLY")
        return [
            self.find_facemask(checkbox_crop),
            self.find_lma(checkbox_crop),
            self.find_ett(checkbox_crop),
            self.find_trach(checkbox_crop),
        ]

    def find_facemask(self, checkbox_crop):
        template_name = "Template - FM.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_fm = checkbox_crop[
                point[1] : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_fm

        return 0

    def find_lma(self, checkbox_crop):
        template_name = "Template - LMA.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_lma = checkbox_crop[
                point[1] + 5 : point[1] + checkbox_height - 5,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_lma

        return 0

    def find_ett(self, checkbox_crop):
        template_name = "Template - ETT.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_ett = checkbox_crop[
                point[1] : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_ett

        return 0

    def find_trach(self, checkbox_crop):
        template_name = "Template - Trach.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_trach = checkbox_crop[
                point[1] : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_trach

        return 0

    # Airway Placement Aid Column
    def find_apa(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(image.crop(column_coordinates["AIRWAY_PLACEMENT_AID"]))
        template_name = "Template - APA.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_apa = checkbox_crop[
                point[1] : point[1] + checkbox_height // 4 + 6,
                point[0] : point[0] + checkbox_width - 65,
            ]
            cropped_box_f = checkbox_crop[
                point[1]
                + checkbox_height // 4
                - 3 : point[1]
                + checkbox_height // 2
                + 3,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_b = checkbox_crop[
                point[1]
                + checkbox_height // 2
                - 3 : point[1]
                + 3 * checkbox_height // 4
                - 5,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_other = checkbox_crop[
                point[1]
                + 3 * checkbox_height // 4
                - 10 : point[1]
                + checkbox_height
                - 10,
                point[0] : point[0] + checkbox_width,
            ]
            return [cropped_box_apa, cropped_box_f, cropped_box_b, cropped_box_other]

        print("STEP 2 ONLY")
        return [
            self.find_apa_used(checkbox_crop),
            self.find_fibroscope(checkbox_crop),
            self.find_bronchoscope(checkbox_crop),
            self.find_other_view(checkbox_crop),
        ]

    def find_apa_used(self, checkbox_crop):
        template_name = "Template - APA Used.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_apa_used = checkbox_crop[
                point[1] : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width - 65,
            ]
            return cropped_box_apa_used

        return 0

    def find_fibroscope(self, checkbox_crop):
        template_name = "Template - Fibroscope.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_f = checkbox_crop[
                point[1] + 5 : point[1] + checkbox_height - 5,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_f

        return 0

    def find_bronchoscope(self, checkbox_crop):
        template_name = "Template - Bronchoscope.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_b = checkbox_crop[
                point[1] + 5 : point[1] + checkbox_height - 5,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_b

        return 0

    def find_other_view(self, checkbox_crop):
        template_name = "Template - OtherView.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_other = checkbox_crop[
                point[1] : point[1] + checkbox_height - 10,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_other

        return 0

    # LRA Column
    def find_lra(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(image.crop(column_coordinates["LRA"]))
        template_name = "Template - LRA.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_lra = checkbox_crop[
                point[1] : point[1] + checkbox_height - 10,
                point[0] : point[0] + checkbox_width - 57,
            ]
            return cropped_box_lra

        return 0

    # Tubes and Lines Column
    def find_tubes_and_lines(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(image.crop(column_coordinates["TUBES_AND_LINES"]))
        template_name = "Template - TnL.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_piv = checkbox_crop[
                point[1] + checkbox_height // 4 : point[1] + checkbox_height // 4 + 14,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_civ = checkbox_crop[
                point[1]
                + checkbox_height // 4
                + 10 : point[1]
                + checkbox_height // 4
                + 25,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_uc = checkbox_crop[
                point[1]
                + checkbox_height // 4
                + 22 : point[1]
                + checkbox_height // 4
                + 37,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_gt = checkbox_crop[
                point[1]
                + checkbox_height // 4
                + 34 : point[1]
                + checkbox_height // 4
                + 49,
                point[0] : point[0] + checkbox_width,
            ]
            return cropped_box_piv, cropped_box_civ, cropped_box_uc, cropped_box_gt

        print("STEP 2 ONLY")
        iv_checkbox = self.find_iv(checkbox_crop)
        tubes = self.find_tubes(checkbox_crop)
        return [
            iv_checkbox[0],
            iv_checkbox[1],
            tubes[0],
            tubes[1],
        ]

    def find_iv(self, checkbox_crop):
        template_name = "Template - IV.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_piv = checkbox_crop[
                point[1] : point[1] + checkbox_height // 2 + 3,
                point[0] : point[0] + checkbox_width,
            ]

            cropped_box_civ = checkbox_crop[
                point[1] + checkbox_height // 2 : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return [cropped_box_piv, cropped_box_civ]

        return [0, 0]

    def find_tubes(self, checkbox_crop):
        template_name = "Template - Tubes.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_uc = checkbox_crop[
                point[1] : point[1] + checkbox_height // 2 + 3,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_gt = checkbox_crop[
                point[1] + checkbox_height // 2 : point[1] + checkbox_height - 3,
                point[0] : point[0] + checkbox_width,
            ]
            return [cropped_box_uc, cropped_box_gt]

        return [0, 0]

    # Monitoring Detail 1 Column
    def find_md1(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(
            image.crop(column_coordinates["MONITORING_DETAILS_LEFT_COLUMN"])
        )
        template_name = "Template - MD1 wide.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, _ = self.get_template_height_and_width(template_name)
            # Using template matching to crop the checkbox
            cropped_box_ecg = checkbox_crop[
                point[1] + 5 : point[1] + 18, point[0] : point[0] + checkbox_width - 5
            ]
            cropped_box_nibp = checkbox_crop[
                point[1] + 15 : point[1] + 28, point[0] : point[0] + checkbox_width - 5
            ]
            cropped_box_spo2 = checkbox_crop[
                point[1] + 28 : point[1] + 40, point[0] : point[0] + checkbox_width - 5
            ]
            cropped_box_etco2 = checkbox_crop[
                point[1] + 40 : point[1] + 53, point[0] : point[0] + checkbox_width - 5
            ]
            cropped_box_steth = checkbox_crop[
                point[1] + 52 : point[1] + 64, point[0] : point[0] + checkbox_width - 5
            ]
            return (
                cropped_box_ecg,
                cropped_box_nibp,
                cropped_box_spo2,
                cropped_box_etco2,
                cropped_box_steth,
            )

        print("STEP 2 ONLY")
        md1_3 = self.find_md1_3(checkbox_crop)
        md1_2 = self.find_md1_2(checkbox_crop)
        return [
            md1_3[0],
            md1_3[1],
            md1_3[2],
            md1_2[0],
            md1_2[1],
        ]

    def find_md1_3(self, checkbox_crop):
        template_name = "Template - MD1.3.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, _ = self.get_template_height_and_width(template_name)
            # Using template matching to crop the checkbox
            cropped_box_ecg = checkbox_crop[
                point[1] + 5 : point[1] + 17, point[0] : point[0] + checkbox_width
            ]
            cropped_box_nibp = checkbox_crop[
                point[1] + 15 : point[1] + 28, point[0] : point[0] + checkbox_width
            ]
            cropped_box_spo2 = checkbox_crop[
                point[1] + 25 : point[1] + 39, point[0] : point[0] + checkbox_width
            ]
            return [cropped_box_ecg, cropped_box_nibp, cropped_box_spo2]

        return [0, 0, 0]

    def find_md1_2(self, checkbox_crop):
        template_name = "Template - MD1.2 narrow.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_etco2 = checkbox_crop[
                point[1] : point[1] + checkbox_height // 2 + 3,
                point[0] : point[0] + checkbox_width,
            ]
            cropped_box_steth = checkbox_crop[
                point[1] + checkbox_height // 2 : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return [cropped_box_etco2, cropped_box_steth]

        return [0, 0]

    # Monitoring Detail 2 Column
    def find_md2(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(
            image.crop(column_coordinates["MONITORING_DETAILS_RIGHT_COLUMN"])
        )
        template_name = "Template - MD2.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, _ = self.get_template_height_and_width(template_name)
            # Using template matching to crop the checkbox
            cropped_box_temp = checkbox_crop[
                point[1] + 2 : point[1] + 16,
                point[0] + 5 : point[0] + checkbox_width - 5,
            ]
            cropped_box_nmt = checkbox_crop[
                point[1] + 15 : point[1] + 27,
                point[0] + 5 : point[0] + checkbox_width - 5,
            ]
            cropped_box_uo = checkbox_crop[
                point[1] + 25 : point[1] + 38,
                point[0] + 5 : point[0] + checkbox_width - 5,
            ]
            cropped_box_abp = checkbox_crop[
                point[1] + 36 : point[1] + 48,
                point[0] + 5 : point[0] + checkbox_width - 5,
            ]
            cropped_box_mdother = checkbox_crop[
                point[1] + 46 : point[1] + 60,
                point[0] + 5 : point[0] + checkbox_width - 5,
            ]
            return [
                cropped_box_temp,
                cropped_box_nmt,
                cropped_box_uo,
                cropped_box_abp,
                cropped_box_mdother,
            ]

        print("STEP 2 ONLY")
        md2_3 = self.find_md2_3(checkbox_crop)
        md2_2 = self.find_md2_2(checkbox_crop)
        return [
            md2_3[0],
            md2_3[1],
            md2_3[2],
            md2_2[0],
            md2_2[1],
        ]

    def find_md2_3(self, checkbox_crop):
        template_name = "Template - MD2.3.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            # I have no clue why, but checkbox width in this method uses a
            # different template to compute checkbox width.
            checkbox_width, _ = self.get_template_height_and_width("Template - MD2.JPG")
            # Using template matching to crop the checkbox
            cropped_box_temp = checkbox_crop[
                point[1] + 4 : point[1] + 15, point[0] : point[0] + checkbox_width - 5
            ]
            cropped_box_nmt = checkbox_crop[
                point[1] + 14 : point[1] + 26, point[0] : point[0] + checkbox_width - 5
            ]
            cropped_box_uo = checkbox_crop[
                point[1] + 25 : point[1] + 37, point[0] : point[0] + checkbox_width - 5
            ]
            return [cropped_box_temp, cropped_box_nmt, cropped_box_uo]

        return [0, 0, 0]

    def find_md2_2(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(
            image.crop(column_coordinates["MONITORING_DETAILS_RIGHT_COLUMN"])
        )
        template_name = "Template - MD2.2.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, checkbox_height = self.get_template_height_and_width(
                template_name
            )
            # Using template matching to crop the checkbox
            cropped_box_abp = checkbox_crop[
                point[1] : point[1] + checkbox_height // 2 + 3,
                point[0] : point[0] + checkbox_width,
            ]

            cropped_box_mdother = checkbox_crop[
                point[1] + checkbox_height // 2 : point[1] + checkbox_height,
                point[0] : point[0] + checkbox_width,
            ]
            return [cropped_box_abp, cropped_box_mdother]

        return 0, 0

    # Patient Position 1 Column
    def find_pp1(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(
            image.crop(column_coordinates["PATIENT_POSITION_LEFT_COLUMN"])
        )
        template_name = "Template - PP1 Narrow.JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, _ = self.get_template_height_and_width(template_name)
            # Using template matching to crop the checkbox
            cropped_box_supine = checkbox_crop[
                point[1] : point[1] + 18, point[0] - 7 : point[0] + checkbox_width + 12
            ]
            cropped_box_prone = checkbox_crop[
                point[1] + 12 : point[1] + 28,
                point[0] - 7 : point[0] + checkbox_width + 12,
            ]
            cropped_box_lithotomy = checkbox_crop[
                point[1] + 22 : point[1] + 39,
                point[0] - 7 : point[0] + checkbox_width + 12,
            ]
            cropped_box_sitting = checkbox_crop[
                point[1] + 33 : point[1] + 50,
                point[0] - 7 : point[0] + checkbox_width + 12,
            ]
            return [
                cropped_box_supine,
                cropped_box_prone,
                cropped_box_lithotomy,
                cropped_box_sitting,
            ]

        return [0, 0, 0, 0]

    # Patient Position 2 column
    def find_pp2(self, checkbox_crop):
        # Crop the image for the correct column
        image = Image.fromarray(checkbox_crop)
        checkbox_crop = np.array(
            image.crop(column_coordinates["PATIENT_POSITION_RIGHT_COLUMN"])
        )
        template_name = "Template - PP2 narrow(remake).JPG"
        # If there is a match, return the best fit
        if self.match_is_found(checkbox_crop, template_name):
            point = self.compute_point(checkbox_crop, template_name)
            checkbox_width, _ = self.get_template_height_and_width(template_name)
            # Using template matching to crop the checkbox
            cropped_box_trendelenburg = checkbox_crop[
                point[1] : point[1] + 18, point[0] - 7 : point[0] + checkbox_width + 12
            ]
            cropped_box_fowler = checkbox_crop[
                point[1] + 12 : point[1] + 28,
                point[0] - 7 : point[0] + checkbox_width + 12,
            ]
            cropped_box_lateral = checkbox_crop[
                point[1] + 22 : point[1] + 39,
                point[0] - 7 : point[0] + checkbox_width + 12,
            ]
            return [cropped_box_trendelenburg, cropped_box_fowler, cropped_box_lateral]

        return [0, 0, 0]
