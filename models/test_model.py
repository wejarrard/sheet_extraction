# Imports
import pandas as pd
import os
import numpy as np
from numpy.distutils.misc_util import is_sequence
from bs4 import BeautifulSoup  # this is to extract info from the xml, if we use it in the end
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import pickle

import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
# from sklearn.metrics import f1_score, precision_score, recall_score
import statistics

import os
from datetime import datetime
from pathlib import Path
from sys import platform

############################ User Parameters ############################
torch.manual_seed(0)

user = "n"
if user == "n":
    computing_id = "na3au"
    xml_ver_string = "html.parser"
elif user == "k":
    computing_id = "kmf4tg"
    xml_ver_string = "xml"

local_mode = True
parallel = True
csv_mode = True
subsample = True
prct = 0.171298804

if local_mode:
    model_string = "2022_02_24-03_02_27_PM_TRAINING/full_model.pt"
    batch_size = 1
    if subsample:
        selfcsv_df = pd.read_csv("../frame_MasterList.csv").head(50)
    else:
        selfcsv_df = pd.read_csv("../frame_MasterList.csv")
    dir_path = os.getcwd()
    labels_txt = pd.read_csv("../2022-02-15.txt", sep=" ", header=None)
    labels_list = labels_txt[0].tolist()
    # labels_list = [int(x) + 1 for x in labels_list]
    num_classes_labels = len(labels_list) + 1
else:
    model_string = "2021_01_04-08_23_03_PM_NOTEBOOK/full_model_25.pt"
    batch_size = 64
    selfcsv_df = pd.read_csv("frame_MasterList.csv")
    dir_path = "/scratch/" + computing_id + "/modelRuns"
    labels_txt = pd.read_csv("../2022-02-15.txt", sep=" ", header=None)
    labels_list = labels_txt[0].tolist()
    # labels_list = [int(x) + 1 for x in labels_list]
    num_classes_labels = len(labels_list) + 1

##########################################################################

#print("Your platform is: ", platform)
if platform == "win32":
    unix = False
else:
    unix = True

if unix:
    # Unix
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    directory = dir_path + "/" + current_time + "_TESTING"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_output_path = directory + "/"
    modelPath = dir_path + "/" + model_string
    print(f'Creation of directory at {directory} successful')
else:
    try:
        # Windows
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        directory = dir_path + "\\" + current_time + "_TESTING"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_output_path = directory + "\\"
        modelPath = dir_path + "\\" + model_string
        print(f'Creation of directory at {directory} successful')
    except:
        print(f'Creation of directory at {directory} failed')

'''if unix:
    print("Unix system detected.")
else:
    print("Windows system detected")
'''
# Generate the target location in the image
def generate_target(image_id, file):
    with open(file) as f:
        lines = [line.rstrip('\n') for line in f]
    num_objs = len(lines)

    boxes = []
    labels = []
    for i in range(num_objs):
        current = lines[i]
        contents = current.split(" ")
        if len(contents) != 5:
            print("Error: Annotation does not contain 4 coordinates or is mising label")
        else:
            img = selfcsv_df.loc[image_id, 'image_path']
            img = Image.open(img).convert("L")
            img_width, img_height = img.size

            class_label = int(contents[0])
            x_center = float(contents[1])
            y_center = float(contents[2])

            width = float(contents[3])
            height = float(contents[4])

            xmin = (x_center - width/2 )  * img_width
            xmax = (x_center + width/2) * img_width
            ymin = (y_center - height/2)  * img_height
            ymax = (y_center + height/2) * img_height

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_label)

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    img_id = torch.tensor([image_id])

    # target for row xyz with 2 annotations =
    # {"image_id": ["XYZ.png"], "boxes": [[c1,c2,c3,c4], [c1,c2,c3,c4]], "labels": [0, 432]}
    target = {}
    target['boxes'] = boxes
    target['labels'] = labels
    target['image_id'] = img_id
    return target


data_transform = transforms.Compose([  # transforms.Resize((320,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.repeat(3,1,1))])


def collate_fn(batch):
    return tuple(zip(*batch))  # will need adjusting when pathing is adjusted


class FullImages(object):
    def __init__(self, transforms=None):
        self.csv = selfcsv_df
        #print(len(self.csv))
        self.imgs = self.csv.image_path.tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)
        # return self.csv_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.csv.loc[idx, 'image_path']
        annotation = self.csv.loc[idx, 'annotation_path']

        img = Image.open(img).convert("L")
        target = generate_target(idx, annotation)

        # label = self.labels[idx]
        # label = OHE(label)
        # label = torch.as_tensor(label, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target



dts = FullImages(data_transform)

data_loader = torch.utils.data.DataLoader(
    dts, batch_size=batch_size, collate_fn=collate_fn)

len_dataloader = len(data_loader)
#print(f'Batches in test dataset: {len_dataloader}')


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        if not param.requires_grad:
            print(name)
    return model

device = torch.device('cpu')  # testing only on CPU
model = get_model_instance_segmentation(num_classes_labels)

state_dict = torch.load(modelPath, map_location=device)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

if parallel == True:
    model = nn.DataParallel(model)

print(f'Model {model_string} loaded.')



def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data
    #print(img.shape)

    ax.imshow(img.permute(1, 2, 0))  # move channel to the end so that the image can be shown accordingly

    #print(img.shape)
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.cpu()
        #print(xmin)

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def plot_images(num):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    img_tensor = imgs[num]
    annotation = annotations[num]
    # for key, value in annotation.items():
    #         print(key, value)
    prediction = preds[num]

    img = img_tensor.cpu().data
    img = img[0, :, :]

    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(img, cmap='gray')

    ix = 0
    for box in annotation["boxes"]:
        # print(annotations[ix])
        xmin, ymin, xmax, ymax = box.tolist()
        value = annotation["labels"][ix]
        img_id = annotation["image_id"].item()
        file_name = selfcsv_df.loc[img_id, :].image_path
        if unix:
            set = file_name.split("/")[7]
            video = file_name.split("/")[8]
            file_name = file_name.split("/")[10]
        else:
            set = file_name.split("\\")[7]
            video = file_name.split("\\")[8]
            file_name = file_name.split("\\")[10]
        file_name = file_name[:-4]
        output_name = set + "_" + video + "_" + file_name
        text = value
        #colors = ["r", "#00FF00", "#0000FF"]
        #rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
        #                         edgecolor=colors[value], facecolor='none')
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
        edgecolor='blue', facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        #ax[0].text(target_x, target_y, text, color=colors[value])
        ax[0].text(target_x, target_y, text, color='blue')
        ax[0].add_patch(rect)
        ix += 1

    ix = 0
    #print(str(len(prediction["boxes"])) + " prediction boxes made for " + str(
    #    len(annotation["boxes"])) + " actual boxes in " + str(output_name))
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        value = prediction["labels"][ix]
        text = value
        colors = ["r", "#00FF00", "#0000FF"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor='blue', facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        ax[1].text(target_x, target_y, text, color=colors[value])
        ax[1].add_patch(rect)
        ix += 1

    # figname = file_name+"_"+input+".png"
    # fig.savefig(figname)
    if local_mode:
        plt.show()

def get_iou(num):
    # BEGIN GET
    annotation = annotations[num]
    prediction = preds[num]
    labels = prediction["labels"]
    annotation_boxes = annotation["boxes"].tolist()
    labels_list = labels.tolist()

    th_better = 0.3
    th_X = 2
    th_Y = 2

    ##### Original prediction boxes
    ix = 0
    voc_iou = []
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        if len(iou_list) != 0:
            max_val = max(iou_list)
            max_val_rounded = round(max(iou_list), 2)
            voc_iou.append(max_val)

        ix += 1

    ##### Calculate accuracy and IoU metrics
    prediction_mod = prediction["boxes"]
    ats_voc_iou_og = []
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for mod_box in prediction_mod:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_box.tolist()
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        if len(iou_list) != 0:
             max_val = max(iou_list)
             ats_voc_iou_og.append(max_val)

    ##### Clustered prediction boxes
    # Collapse predictions
    prediction_mod = prediction["boxes"].tolist()
    subset_indices = []
    c_ix = 0
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        collapsed = False
        for compare_box in prediction["boxes"]:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box.tolist()

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (
                    ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    # print("pred:",len(prediction_mod))
    for index_num in subset_indices:
        prediction_mod.pop(index_num)
        labels_list.pop(index_num)
        # print(labels_list)
    # print("pred:",len(prediction_mod))
    prediction_superset = []
    super = []
    # prediction_mod = prediction["boxes"]
    bb = 0
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > th_better:
                    if (xmin + mod_xmin) / th_X < xmin:
                        xmin = (xmin + mod_xmin) / th_X
                    if (ymin + mod_ymin) / th_Y < ymin:
                        ymin = (ymin + mod_ymin) / th_Y
                    if (xmax + mod_xmax) / th_X > xmax:
                        xmax = (xmax + mod_xmax) / th_X
                    if (ymax + mod_ymax) / th_Y > ymax:
                        ymax = (ymax + mod_ymax) / th_Y

                    prediction_superset[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset.append([xmin, ymin, xmax, ymax])
            super.append(labels_list[bb])
        bb+=1

    ##### SUPERSET
    # print("b4 pred:",len(prediction_mod))
    prediction_mod = prediction_superset
    labels_list = super
    # print("ater pred:",len(prediction_mod))
    #print(prediction_superset)
    subset_indices = []
    c_ix = 0
    for box in prediction_superset:
        xmin, ymin, xmax, ymax = box

        collapsed = False
        for compare_box in prediction_superset:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (
                    ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    # print("pred:",len(prediction_mod))
    for index_num in subset_indices:
        prediction_mod.pop(index_num)
        labels_list.pop(index_num)
        print(labels_list)
    # print("pred:",len(prediction_mod))
    prediction_superset_clustered = []
    # prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset_clustered:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > 0.8:
                    if (xmin + mod_xmin) / th_X < xmin:
                        xmin = (xmin + mod_xmin) / th_X
                    if (ymin + mod_ymin) / th_Y < ymin:
                        ymin = (ymin + mod_ymin) / th_Y
                    if (xmax + mod_xmax) / th_X > xmax:
                        xmax = (xmax + mod_xmax) / th_X
                    if (ymax + mod_ymax) / th_Y > ymax:
                        ymax = (ymax + mod_ymax) / th_Y

                    prediction_superset_clustered[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset_clustered.append([xmin, ymin, xmax, ymax])

    prediction_mod = prediction_superset_clustered
    subset_indices = []
    c_ix = 0
    for box in prediction_superset_clustered:
        # print(box)
        # print(c_ix)
        xmin, ymin, xmax, ymax = box
        p_area = (xmax - xmin + 1) * (ymax - ymin + 1)

        collapsed = False
        for compare_box in prediction_superset_clustered:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area)

            if iou > 0.8 and iou != 1 and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)

    # print("pred:",len(prediction_mod))
    for index_num in subset_indices:
        prediction_mod.pop(index_num)
        labels_list.pop(index_num)
        # print(labels_list)
    # print("pred:",len(prediction_mod))
    # prediction_mod = prediction_mod[1:]

    ix = 0
    voc_iou_mod = []
    voc_height = []
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box

        iou_list = []
        pred_csv, val_csv = [], []
        iou_heights = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

            max_y, min_y = max(yA, yB), min(yA, yB)
            pred_height = abs(max_y - min_y)
            iou_heights.append(pred_height)

        if len(iou_list) != 0:
            max_val = max(iou_list)
            max_val_ix = iou_list.index(max_val)
            voc_iou_mod.append(max_val)
            voc_height.append(iou_heights[max_val_ix])


        ix += 1

    ##### Calculate accuracy and IoU metrics
    ats_voc_iou_mod = []
    ats_height = []
    tpfp = []
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        iou_heights = []
        for mod_box in prediction_mod:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

            max_y, min_y = max(yA, yB), min(yA, yB)
            pred_height = abs(max_y - min_y)
            iou_heights.append(pred_height)

        if len(iou_list) != 0:
            # print(iou_list)
            max_val = max(iou_list)
            max_val_ix = iou_list.index(max_val)
            ats_voc_iou_mod.append(max_val)
            ats_height.append(iou_heights[max_val_ix])

    #print("\n Original Predictions")
    #print(f'{len(prediction["boxes"])} boxes made for {len(annotation["boxes"])} actual boxes in {str(output_name)} for {identifier} with note {input} (INDEX {num})')
    if len(voc_iou) == 0:
        mean_iou = 0
        #print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        og_mean_iou = sum(voc_iou) / len(voc_iou)
        og_accuracy = sum([1 if entry >= 0.4 else 0 for entry in ats_voc_iou_og]) / len(prediction["boxes"])
    #print("\n Clustered Predictions")
    if len(voc_iou_mod) == 0:
        mean_iou = 0
        print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        for item in voc_iou_mod:
            if item == 0:
                tpfp.append("FP")
            else:
                tpfp.append("TP")

        mean_iou = sum(voc_iou_mod) / len(voc_iou_mod)
        fp = voc_iou_mod.count(0)
        bp = sum((i > 0 and i < 0.4) for i in voc_iou_mod)
        gp = sum((i >= 0.4) for i in voc_iou_mod)
        fn = sum((i == 0) for i in ats_voc_iou_mod)
        accuracy = sum([1 if entry >= 0.4 else 0 for entry in ats_voc_iou_mod]) / len(voc_iou_mod)
        if fp!=0:
            precision = (gp + bp) / (gp + bp + fp)
        else:
            precision = 1
        if (gp + bp + fn) != 0:
            recall = (gp + bp) / (gp + bp + fn)
        else:
            recall = 0
        if (precision + recall) !=0:
            f1 = (2 * precision * recall) / (precision + recall)
            if f1 > 1:
                print("gp",gp)
                print("bp", bp)
                print("fn", fn)
                print("fp", fp)
        else:
            f1 = 1
        voc_tp_height_indices = [idx for idx, element in enumerate(voc_height) if element > 0]
        voc_tp_height = [voc_height[i] for i in voc_tp_height_indices]
        voc_fn_height_indices = [idx for idx, element in enumerate(ats_height) if element == 0]
        voc_fn_height = [ats_height[i] for i in voc_fn_height_indices]
        value = labels_list
        true_value = annotation['labels'].tolist()
        pred_csv.append(value)
        val_csv.append(true_value)
    if len(voc_iou_mod) == 0 and len(voc_iou) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif len(voc_iou) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif len(voc_iou_mod) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        return [accuracy, mean_iou, og_accuracy, og_mean_iou, f1, pred_csv, val_csv, voc_tp_height, voc_fn_height, tpfp]
    # END GET

def plot_iou(num, input):
    # BEGIN PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2)

    identifier = "test"
    img_tensor = imgs[num]
    annotation = annotations[num]
    prediction = preds[num]
    labels = prediction["labels"]
    labels = labels.tolist()
    th_better = 0.3
    th_X = 2
    th_Y = 2
    #print(prediction["boxes"])

    img = img_tensor.cpu().data
    # print(f'img is {img.shape}')
    # img = img.permute(1, 2, 0)
    # print(f'img is {img.shape}')

    img = img[0, :, :]
    annotation_boxes = annotation["boxes"].tolist()

    ##### Subplot 1: Original prediction boxes
    ax1.imshow(img, cmap='gray')

    ix = 0
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        value = annotation["labels"][ix]
        img_id = annotation["image_id"].item()
        file_name = selfcsv_df.loc[img_id, :].image_path
        if unix:
            set = file_name.split("/")[7]
            video = file_name.split("/")[8]
            file_name = file_name.split("/")[10]
        else:
            set = file_name.split("\\")[7]
            video = file_name.split("\\")[8]
            file_name = file_name.split("\\")[10]
        file_name = file_name[:-4]
        output_name = set + "_" + video + "_" + file_name + "_" + identifier
        text = value
        colors = ["r", "r", "r"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor='blue', facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        #ax1.text(target_x, target_y, text, color=colors[value])
        ax1.add_patch(rect)
        ix += 1

    ix = 0
    voc_iou = []
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        if len(iou_list) != 0:
            max_val = max(iou_list)
            max_val_rounded = round(max(iou_list), 2)
            voc_iou.append(max_val)

            value = prediction["labels"][ix]

            #text = json.dumps(map_dict)
            text = max_val_rounded
            colors = ["r", "#00FF00", "#0000FF"]
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                     edgecolor=colors[value], facecolor='none')
            target_x = xmin
            target_y = ymin - 5
            #ax1.text(target_x, target_y, text, color=colors[value])
            ax1.add_patch(rect)
        ix += 1

    ##### Calculate accuracy and IoU metrics
    prediction_mod = prediction["boxes"]
    ats_voc_iou_og = []
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for mod_box in prediction_mod:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_box.tolist()
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        if len(iou_list) != 0:
            max_val = max(iou_list)
            ats_voc_iou_og.append(max_val)

    ##### Subplot 2: Clustered prediction boxes
    ax2.imshow(img, cmap='gray')

    ix = 0
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        value = annotation["labels"][ix]
        img_id = annotation["image_id"].item()
        file_name = selfcsv_df.loc[img_id, :].image_path
        if unix:
            set = file_name.split("/")[7]
            video = file_name.split("/")[8]
            file_name = file_name.split("/")[10]
        else:
            set = file_name.split("\\")[7]
            video = file_name.split("\\")[8]
            file_name = file_name.split("\\")[10]
        file_name = file_name[:-4]
        output_name = set + "_" + video + "_" + file_name + "_" + identifier
        text = value
        colors = ["r", "r", "r"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor='blue', facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        #ax2.text(target_x, target_y, text, color=colors[value])
        ax2.add_patch(rect)
        ix += 1

    # Collapse predictions
    prediction_mod = prediction["boxes"].tolist()
    subset_indices = []
    c_ix = 0
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        collapsed = False
        for compare_box in prediction["boxes"]:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box.tolist()

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)
        labels.pop(index_num)
        print(labels)

    prediction_superset = []
    #prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > th_better:
                    if (xmin + mod_xmin)/th_X < xmin:
                        xmin = (xmin + mod_xmin)/th_X
                    if (ymin + mod_ymin)/th_Y < ymin:
                        ymin = (ymin + mod_ymin)/th_Y
                    if (xmax + mod_xmax)/th_X > xmax:
                        xmax = (xmax + mod_xmax)/th_X
                    if (ymax + mod_ymax)/th_Y > ymax:
                        ymax = (ymax + mod_ymax)/th_Y

                    prediction_superset[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset.append([xmin, ymin, xmax, ymax])

    ##### SUPERSET
    prediction_mod = prediction_superset
    #print(prediction_superset)
    subset_indices = []
    c_ix = 0
    for box in prediction_superset:
        xmin, ymin, xmax, ymax = box

        collapsed = False
        for compare_box in prediction_superset:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (
                    ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)
        labels.pop(index_num)

    prediction_superset_clustered = []
    # prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset_clustered:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > 0.8:
                    if (xmin + mod_xmin)/th_X < xmin:
                        xmin = (xmin + mod_xmin)/th_X
                    if (ymin + mod_ymin)/th_Y < ymin:
                        ymin = (ymin + mod_ymin)/th_Y
                    if (xmax + mod_xmax)/th_X > xmax:
                        xmax = (xmax + mod_xmax)/th_X
                    if (ymax + mod_ymax) / th_Y > ymax:
                        ymax = (ymax + mod_ymax) / th_Y

                    prediction_superset_clustered[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset_clustered.append([xmin, ymin, xmax, ymax])

    prediction_mod = prediction_superset_clustered
    subset_indices = []
    c_ix = 0
    for box in prediction_superset_clustered:
        xmin, ymin, xmax, ymax = box
        p_area = (xmax - xmin + 1) * (ymax - ymin + 1)

        collapsed = False
        for compare_box in prediction_superset_clustered:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area)

            if iou > 0.8 and iou != 1 and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)
        labels.pop(index_num)

    # prediction_mod = prediction_mod[1:]

    ix = 0
    voc_iou_mod = []
    voc_height = []
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box

        iou_list = []
        pred_csv, val_csv = [], []
        iou_heights = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)

            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

            max_y, min_y = max(yA, yB), min(yA, yB)
            pred_height = abs(max_y - min_y)
            iou_heights.append(pred_height)

        if len(iou_list) != 0:
            max_val = max(iou_list)
            max_val_ix = iou_list.index(max_val)
            voc_iou_mod.append(max_val)
            voc_height.append(iou_heights[max_val_ix])

            value = labels[ix]
            true_value = annotation["labels"][max_val_ix]
            pred_csv.append(value)
            val_csv.append(true_value)

            #text = json.dumps(map_dict)
            colors = ["r", "#00FF00", "#0000FF"]
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                     edgecolor=colors[value], facecolor='none')
            target_x = xmin
            target_y = ymin - 5
            '''if text == 0.0:
                pass
            else:
                ax2.text(target_x, target_y, text, color=colors[value])'''
            ax2.add_patch(rect)
        ix += 1

    ##### Calculate accuracy and IoU metrics
    ats_voc_iou_mod =[]
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for mod_box in prediction_mod:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        if len(iou_list) != 0:
            max_val = max(iou_list)
            ats_voc_iou_mod.append(max_val)

    print("\n Original Predictions")
    print(f'{len(prediction["boxes"])} boxes made for {len(annotation["boxes"])} actual boxes in {str(output_name)} for {identifier} with note {input} (INDEX {num})')
    if len(voc_iou) == 0:
        mean_iou = 0
        print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        mean_iou = sum(voc_iou) / len(voc_iou)
        fp = voc_iou.count(0)
        bp = sum((i > 0 and i < 0.4) for i in voc_iou)
        gp = sum((i >= 0.4) for i in voc_iou)
        accuracy = sum([1 if entry >= 0.4 else 0 for entry in ats_voc_iou_og]) / len(annotation["boxes"])
        print(f'{fp} false positives (IOU = 0)')
        print(f'{bp} bad positives (0 < IOU < 0.4)')
        print(f'{gp} good positives (IOU >= 0.4)')
        print(f'Mean IOU: {mean_iou}')
        print(f'Accuracy: {accuracy*100}%')
        #print(f'Predictions for Image {num} have mean IOU: {mean_iou} and accuracy: {accuracy}')

    print("\n Clustered Predictions")
    if len(voc_iou_mod) == 0:
        mean_iou = 0
        print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        mean_iou = sum(voc_iou_mod) / len(voc_iou_mod)
        # fp = voc_iou_mod.count(0)
        # bp = sum((i > 0 and i < 0.4) for i in voc_iou_mod)
        # gp = sum((i >= 0.4) for i in voc_iou_mod)
        # fn = sum((i == 0) for i in ats_voc_iou_mod)
        # accuracy = sum([1 if entry >= 0.4 else 0 for entry in ats_voc_iou_mod]) / len(voc_iou_mod)
        # if fp != 0:
        #     precision = (gp+bp)/fp
        # else:
        #     precision = 1
        # recall = (gp+bp)/(gp+bp+fn)
        # f1 = (2*precision*recall)/(precision+recall)
        # print(f'{len(prediction_mod)} boxes made for {len(annotation["boxes"])} actual boxes in {str(output_name)} for {identifier} with note {input} (INDEX {num})')
        # print(f'{fp} false positives (IOU = 0)')
        # print(f'{bp} bad positives (0 < IOU < 0.4)')
        # print(f'{gp} good positives (IOU >= 0.4)')
        # print(f'Mean IOU: {mean_iou}')
        # print(f'Accuracy: {accuracy*100}%')
        # print(f'F1 Score: {f1*100}%')
        #print(f'Predictions for Image {num} have mean IOU: {mean_iou} and accuracy: {accuracy}')

    # plt.show()
    figname = output_name + "_" + input + ".png"
    fig.savefig(file_output_path + figname)
    # print(f'Figure {figname} saved to {directory}.')
    # END PLOT

# print("Predicted:")
# for i in range(len(preds) - 1):
#     #print(preds[i])
#     plot_image(imgs[i], preds[i])
#     #plot_images(i, f"Input {i}")

iou = list()
acc = list()
og_iou_list = list()
og_acc_list = list()
f1_list = list()
preds_list = list()
vals_list = list()
tp_list = list()
fn_list = list()
tpp_list = list()
if csv_mode:
    iou_df_test = pd.DataFrame(columns=["Clustered_Accuracy", "Clustered_IOU", "Unclustered_Accuracy", "Unclustered_IOU", "Clustered F1","Predictions", "True", "TP Height", "FN Height", "TP or FP"])
    iou_df_test_name = "full_iou_TEST.csv"

with torch.no_grad():
    i = 0
    # Index, max value
    max_i_og = [0, 0]
    max_i_mod = [0, 0]
    for imgs, annotations in data_loader:
        #print(f'Iteration {i}')
        preds = model(imgs)
        #[accuracy, mean_iou, og_accuracy, og_mean_iou, f1, pred_csv, val_csv, voc_tp_height, voc_fn_height]
        accuracy, io, og_acc, og_iou, f1_val, predvals, truevals, tp_y, fn_y, tpp = get_iou(0)
        iou.append(io)
        acc.append(accuracy)
        og_iou_list.append(og_iou)
        og_acc_list.append(og_acc)
        f1_list.append(f1_val)
        preds_list.append(predvals)
        vals_list.append(truevals)
        tp_list.append(tp_y)
        fn_list.append(fn_y)
        tpp_list.append(tpp)
        if csv_mode:
            if max_i_og[1] < og_acc:
                max_i_og[0] = i
                max_i_og[1] = og_acc

            if max_i_mod[1] < accuracy:
                max_i_mod[0] = i
                max_i_mod[1] = accuracy

            len_df = len(iou_df_test)
            iou_df_test.loc[len_df, :] = [accuracy, io, og_acc, og_iou, f1_val, predvals, truevals, tp_y, fn_y, tpp]

            if abs(accuracy) > .9 and i > 3000:
                plot_iou(0,str(accuracy))
            # if i == 268 or i == 269 or i == 270 or i == 0 or i == 1:
            #     plot_iou(0,str(accuracy))

            try:
                if i % 1000 == 0:
                    partial_name = "partial_iou_TEST_" + str(i) + "_images.csv"
                    iou_df_test.to_csv(file_output_path + iou_df_test_name, index=False)
                    print(f'Partial test IOUs for {len(iou_df_test)} images saved to {directory}.')
            except:
                pass
        i+=1
        print(i)
        plt.close()

    mean_acc = np.mean(acc)
    mean_iou = np.mean(iou)
    mean_og_acc = np.mean(og_acc_list)
    mean_og_iou = np.mean(og_iou_list)

if  csv_mode:
    iou_df_test.to_csv(file_output_path + iou_df_test_name, index=False)
    print(f'Full test IOUs for {len(iou_df_test)} images saved to {directory}.')
    print(iou_df_test.sort_values(by='Clustered_IOU', ascending=False).head(5))

    max_test_og = iou_df_test[iou_df_test['Clustered_IOU'] == max_i_og[1]].index.tolist()[0]
    plot_iou(max_test_og, "best_original_acc")

    max_test_mod = iou_df_test[iou_df_test['Clustered_Accuracy'] == max_i_mod[1]].index.tolist()[0]
    plot_iou(max_test_mod, "best_clustered_acc")

print("\n")
print("clustered accuracy: ", mean_acc * 100)
print("clustered iou: ", mean_iou * 100)

print("unclustered accuracy: ", mean_og_acc * 100)
print("unclustered iou: ", mean_og_iou * 100)
