import torch
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import cv2

num_to_drug_dict = {
    1: "atropine",
    2: "bupivacaine",
    3: "cefazolin",
    4: "cefotaxime",
    5: "ceftriaxone",
    6: "cisatracurium",
    7: "dexamethasone",
    8: "diclofenac",
    9: "ephedrine",
    10: "fentanyl",
    11: "hydrocortisone",
    12: "ketamine",
    13: "lidocaine",
    14: "marcaine",
    15: "metronidazole",
    16: "midazolam",
    17: "morphine",
    18: "narcaine",
    19: "neostigmine",
    20: "nimbex",
    21: "norcuron",
    22: "ondansetron",
    23: "paracetamol",
    24: "propofol",
    25: "suxamethonium",
    26: "temp",
    27: "thiopental",
    28: "unknown",
    29: "vecuronium",
    30: "xylocaine",
}


class drug_extractor:
    """drug_extractor - reads drugs from intraoperative images."""

    # todo: some kind of loading to gpu?

    def __init__(
        self,
        path_to_flowsheet: str,
        path_to_YOLO_weights: str,
        path_to_VGG16_weights: str,
        debug: bool = False,
    ):
        """
        Initializes the model.

        Parameters
        ----------
        drug_section_image : PIL Image
            The PIL image of the already cropped drug section.
        path_to_model_weights : str
            path to the yolov5 model weights.
        """
        self.YOLO_model = self.load_YOLO_model(path_to_YOLO_weights)
        self.VGG16_model = self.load_VGG16_model(path_to_VGG16_weights)
        self.drug_section_image = self.crop_drugs(path_to_flowsheet)
        self.detection_results = self.predict_drug_locations(
            self.drug_section_image, debug
        )
        self.predict_drug_values(self.detection_results)
        print(self.detection_results)

    def predict_drug_locations(
        self, drug_section_image, debug: bool
    ) -> pd.core.frame.DataFrame:
        """Uses the YOLOv5 model to make predictions.
        
        Parameters
        ----------
        drug_section_image : PIL image
            Filepath to the image to predict on.

        debug : bool
            Determines whether or not to display the image.

        Returns
        -------
        models.common.Detections object with the model results.
        """
        image = drug_section_image
        results = self.YOLO_model(image)

        result_df = pd.DataFrame(results.pandas().xyxy[0])
        result_df.sort_values(by="confidence", inplace=True)
        if debug:
            results.show()
            results.save()
            print(result_df)

        return result_df

    def predict_drug_values(self, crops_df: pd.core.frame.DataFrame):
        for i in range(len(crops_df)):
            crop = self.drug_crop.crop(
                (
                    crops_df.loc[i].xmin,
                    crops_df.loc[i].ymin,
                    crops_df.loc[i].xmax,
                    crops_df.loc[i].ymax,
                )
            ).save("crop.jpg")
            print(self.CNN_predict("crop.jpg"))

    def load_YOLO_model(self, path_to_YOLO_weights: str):
        """Loads the yolov5 model into memory.
        
        Parameters
        ----------
        path_to_model_weights : str
            The path to the model weights for the YOLOv5 model trained on the cropped
            drug section images.
        
        Returns
        -------
        The YOLOv5 model for detecting drugs.
        """
        # todo: find out how to load the yolov5x.pt file locally
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=path_to_YOLO_weights
        )
        return model

    def load_VGG16_model(self, path_to_VGG16_weights):
        """Loads the CNN for classifying the images

        Parameters
        ----------
        path_to_CNN_weights : str
            The filepath to the weights for the CNN

        Returns
        -------
        The CNN model.
        """
        return tf.keras.models.load_model(path_to_VGG16_weights)

    def CNN_predict(self, filepath_to_image: str) -> list:
        """Makes a prediction using the CNN on a new image.

        Parameters
        ----------
        filepath_to_image : str
            The filepath to the image to generate predictions for.

        Returns
        -------
        A dictionary of predictions.
        """
        image = tf.keras.preprocessing.image.load_img(
            filepath_to_image, target_size=(32, 128)
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = tf.keras.applications.vgg16.preprocess_input(image)
        pred = self.VGG16_model.predict(image)
        return num_to_drug_dict[np.argmax(pred[0])]

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

    def crop_drugs(self, image_path: str):
        """Crops the IV drug section out of the flowsheet
        
        Note: this function was written by the 2021 capstone team with no documentation other than
              the inline comments, so if you want to understand how this works read their paper or
              search for more information about ORB template matching.

        Parameters
        ----------
        image_path : str
            The path to the image.

        Returns
        -------
        An np array with the image data.
        """
        refFilename = (
            "../templates/IntraoperativeForm.JPG"  # This may be in another location.
        )
        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
        refFileWidth = imReference.shape[1]
        im_patient_org = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im_patient = self.image_resize(image=im_patient_org, width=refFileWidth)

        # Registered image will be restored in img.
        # img = aligned image (cv2)
        # image = aligned image (PIL)
        # The estimated homography will be stored in h.
        img, _ = self.align_images(
            im_patient, imReference, max_features=500, good_match_percent=0.1
        )
        image = Image.fromarray(img)
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
        x_min_6 = min_x + int((0 / max_x) * x_dim)
        x_max_6 = min_x + int((190 / max_x) * x_dim)
        y_min_6 = min_y + int((0 / max_y) * y_dim)
        y_max_6 = min_y + int((155 / max_y) * y_dim)

        drug_crop = Image.fromarray(
            np.array(image.crop((x_min_6, y_min_6, x_max_6, y_max_6 + 15)))
        )
        self.drug_crop = drug_crop
        return drug_crop

