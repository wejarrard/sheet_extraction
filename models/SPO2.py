# Imports
#!pip install torchvision
import matplotlib

import pandas as pd
import os
from sys import platform
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

import os
from datetime import datetime


#local, n, or k
user = "local"
rgb_mode = False # False for greyscale
iou_mode = True # False if not plotting iou
number_workers = 2 #min 8 cores
#number_workers = 4 #min 12 - 16 cores
learning_rate = 0.001
weight_decay_rate =  0
early_stop = False

print(f"Learning rate is: {learning_rate}")
print(f"Weight decay is: {weight_decay_rate}")

save_epochs_every = True
save_epochs_num = 25

if save_epochs_every:
    print(f"Partial models will be saved every {save_epochs_num} epochs")

if user == "n":
    computing_id = "na3au"
    rivanna_mode = True
elif user == "k":
    computing_id = "kmf4tg"
    rivanna_mode = True
elif user == "local":
    computing_id = "local"
    rivanna_mode = False

# Check if on windows or mac
if platform == "win32":
    unix = False
else:
    unix = True

# Check if on rivanna and use right variables
print(f"User mode is {computing_id}")

if not rivanna_mode:
    batch_size = 3
    num_epochs = 50
    selfcsv_df = pd.read_csv("../frame_MasterList.csv")

    labels_txt = pd.read_csv("../2022-02-21.txt", sep=" ", header=None)
    labels_list = labels_txt[0].tolist()
    #labels_list = [int(x) + 1 for x in labels_list]
    num_classes_labels = len(labels_list) + 50

    dir_path = os.getcwd()
    xml_ver_string = "xml"
else:
    batch_size = 5
    num_epochs = 3
    selfcsv_df = pd.read_csv("../frame_MasterList.csv") #.head(50)

    labels_txt = pd.read_csv("../2022-02-21.txt", sep=" ", header=None)
    labels_list = labels_txt[0].tolist()
    #labels_list = [int(x) + 1 for x in labels_list]
    num_classes_labels = len(labels_list) + 1

    dir_path = "/scratch/" + computing_id + "/modelRuns"
    xml_ver_string = "html.parser"
    matplotlib.use('Agg')

print(f"Number labels {num_classes_labels}")

try:
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    directory = dir_path + "/" + current_time + "_TRAINING"
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f'Creation of directory at {directory} successful')
except:
    print(f'Creation of directory at {directory} failed')
file_output_path = directory + "/"

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

class FullImages(object):
    def __init__(self, transforms=None):
        self.csv = selfcsv_df
        self.csv_len = self.csv.shape[1]
        self.imgs = self.csv.image_path.tolist()
        self.imgs_len = len(self.imgs)
        self.transforms = transforms

    def __len__(self):
        return self.imgs_len
        # return self.csv_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.csv.loc[idx, 'image_path']
        annotation = self.csv.loc[idx, 'annotation_path']

        img = Image.open(img).convert("L")
        target = generate_target(idx, annotation)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

# Normalize
if rgb_mode:
    data_transform = transforms.Compose([  # transforms.Resize((80,50)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
else:
    data_transform = transforms.Compose([  # transforms.Resize((80,50)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

# Collate/zip images
def collate_fn(batch):
    return tuple(zip(*batch))

dataset = FullImages(data_transform)
data_size = len(dataset)
print(f'Length of Dataset: {data_size}')

indices = list(range(data_size))
test_split = 0.1
split = int(np.floor(test_split * data_size))
print(f'Length of Split Dataset: {split}')

train_indices, test_indices = indices[split:], indices[:split]
len_train_ind, len_test_ind = len(train_indices), len(test_indices)
print(f'Length of Train Instances: {len_train_ind}; Length of Test Instances: {len_test_ind}')

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_fn,
    num_workers=number_workers
)

len_dataloader = len(data_loader)
data_loader_test = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler,
                                               collate_fn=collate_fn, num_workers=number_workers)
len_testdataloader = len(data_loader_test)
print(f'Length of Test: {len_testdataloader}; Length of Train: {len_dataloader}')

# Instance segmentation is crucial in using the full images
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)
    for name, param in model.named_parameters():
        param.requires_grad_(True)
    return model

model = get_model_instance_segmentation(num_classes_labels)

# Check if GPU
cuda = torch.cuda.is_available()
if cuda:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device = torch.device("cuda:0")
        model = nn.DataParallel(model)
    else:
        device = torch.device("cuda")
        print(f'Single CUDA.....baby shark doo doo doo')
else:
    device = torch.device("cpu")
    print(f'But I\'m just a poor CPU and nobody loves me :(')

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr = learning_rate, weight_decay = weight_decay_rate)

tot_ats = 0
epochs = 0

epoch_iou_list = []
epoch_acc_list = []
epoch_losses = []

save_epoch = False
lr_threshold = 0.001

def get_iou(num, ges, ann):
    annotation = ann[num]
    prediction = ges[num]
    annotation_boxes = annotation["boxes"].tolist()

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

    if len(voc_iou) == 0:
        mean_iou = 0
        og_accuracy = 0
        og_mean_iou = 0
        print(f'No predictions made so Mean IOU: {mean_iou}')
    else:

        og_mean_iou = sum(voc_iou) / len(voc_iou)
        fp = voc_iou.count(0)
        bp = sum((i > 0 and i < 0.4) for i in voc_iou)
        gp = sum((i >= 0.4) for i in voc_iou)
        og_accuracy = sum([1 if entry >= 0.4 else 0 for entry in voc_iou]) / len(annotation["boxes"])
        print(f'{len(prediction["boxes"])} boxes made for {len(annotation["boxes"])} actual boxes for (INDEX {num})')
        print(f'{fp} false positives (IOU = 0), {bp} bad positives (0 < IOU < 0.4), {gp} good positives (IOU >= 0.4)')
        #print(f'Mean IOU: {og_mean_iou}')
        #print(f'Accuracy: {og_accuracy*100}%')
        #print(f'Predictions for Image {num} have mean IOU: {og_mean_iou} and accuracy: {
        # og_accuracy}')

    return [og_accuracy, og_mean_iou]

def plot_iou(num, preds, annotations, input):
    # BEGIN PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2)

    identifier = "test"
    img_tensor = imgs[num]
    annotation = annotations[num]
    prediction = preds[num]
    labels = prediction["labels"]
    labels = labels.tolist()

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
        file_name = file_name[:-4]
        output_name = file_name + "_" + identifier
        text = value
        colors = ["r", "r", "r"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor='blue', facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        #ax1.text(target_x, target_y, text, color=colors[value])
        ax1.add_patch(rect)
        ix += 1

    if not rivanna_mode:
        plt.show()

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
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                     edgecolor="blue", facecolor='none')
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
    if len(voc_iou) == 0:
        mean_iou = 0
        print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        mean_iou = sum(voc_iou) / len(voc_iou)
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

    plt.show()
    figname = output_name + "_" + input + ".png"
    if not rivanna_mode:
        fig.savefig(file_output_path + figname)
    # print(f'Figure {figname} saved to {directory}.')
    # END PLOT

#wandb.watch(model)
for epoch in range(num_epochs):

    epochs += 1

    print(f'Epoch: {epochs}')

    model.train()

    epoch_loss = 0
    epoch_iou = 0

    i = 0

    for train_imgs, train_annotations in data_loader:
        # torch.cuda.empty_cache()

        imgs = list(img.to(device) for img in train_imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in train_annotations]

        '''print(f"Length imgs: {len(imgs)}")
        print(f"Length annotations: {len(annotations)}")

        print(f"Length [imgs[0]]: {len([imgs[0]])}")
        print(f"Length [annotations[0]]: {len([annotations[0]])}")
        print(annotations)

        print(f"Length imgs[0]: {len(imgs[0])}")
        print(f"Length annotations[0]: {len(annotations[0])}")'''

        loss_dict = model([imgs[0]], [annotations[0]])
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()

        '''for param_group in optimizer.param_groups:
            if param_group['lr'] < lr_threshold:
                early_stop = True

            if not early_stop:
                pass
                #print(f"Learning rate for epoch {epoch} is {param_group['lr']}")
            else:
                save_epoch = True'''

        losses.backward()
        optimizer.step()

        i += 1
        tot_ats += 1

        epoch_loss += losses.item()

        print(f'Iteration Number: {i}/{len_dataloader}, Loss: {losses}')

    mean_epoch_loss = epoch_loss / i
    epoch_losses.append(mean_epoch_loss)
    #wandb.log({'loss': mean_epoch_loss})

    # Epoch-wise Training IoU
    try:
        if iou_mode:
            model.eval()
            with torch.no_grad():
                for test_imgs, test_annotations in data_loader_test:
                    imgs_test = list(img_test.to(device) for img_test in test_imgs)
                    annotations_test = [{k: v.to(device) for k, v in t.items()} for t in test_annotations]

                guess = model(imgs_test)
                plot_image = plot_iou(0, guess, annotations_test, str(epoch))
                epoch_iou = get_iou(0, guess, annotations_test)

                model.train()

                epoch_acc = epoch_iou[0]
                epoch_avg = epoch_iou[1]
                epoch_iou_list.append(epoch_avg)
                epoch_acc_list.append(epoch_acc)
            print(f"Epoch {epochs} IoU: ", epoch_avg)
    except Exception as e:
        print(e)
        epoch_iou_list.append("Exception")
        pass

    if save_epochs_every and epochs % save_epochs_num == 0:
        if iou_mode:
            df = pd.DataFrame({'Mean_Epoch_Loss': epoch_losses, 'Mean_Training_IOU': epoch_iou_list, 'Mean Accuracy': epoch_acc_list})
        else:
            df = pd.DataFrame({'Mean_Epoch_Loss': epoch_losses})

        partial_name = "partial_model_" + str(epochs)

        try:
            # Save model
            torch.save(model.state_dict(), file_output_path + partial_name + ".pt")
            print(f'Partial model trained on {epochs} epochs saved to {directory}.')
        except:
            print(f'Could not save partial model at epoch {epochs}.')
            pass

        try:
            # Save training metrics
            df.to_csv(file_output_path + partial_name + "_losses.csv", index=False)
            print(f'Partial model metrics trained on {epochs} epochs saved to {directory}.')
        except:
            print(f'Could not save partial model metrics at epoch {epochs}.')
            pass

try:
    # Save training metrics
    full_name = "full_model_losses_" + str(epochs) + ".csv"

    if iou_mode:
        df = pd.DataFrame({'Mean_Epoch_Loss': epoch_losses, 'Mean_Training_IOU': epoch_iou_list, 'Mean Accuracy': epoch_acc_list})
    else:
        df = pd.DataFrame({'Mean_Epoch_Loss': epoch_losses})

    df.to_csv(file_output_path + full_name, index=False)
    print(f'Full model losses for {epochs} epochs saved to {directory}.')
except:
    print("Error with saving CSV")

try:
    # Save model
    torch.save(model.state_dict(), file_output_path + 'full_model.pt')
    print(f'Full model trained on {epochs} epochs saved to {directory}.')
except:
    pass

def plot_annotations(idx):
    fig, ax = plt.subplots(1)
    img = selfcsv_df.loc[idx, 'image_path']
    annotation = selfcsv_df.loc[idx, 'annotation_path']

    img = Image.open(img).convert("L")
    #img = img_tensor.cpu().data
    #img = img[0, :, :]
    ax.imshow(img, cmap='gray')

    target = generate_target(idx, annotation)

    boxes = target["boxes"]
    labels = target["labels"]

    i = 0
    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        i += 1

    if not rivanna_mode:
        plt.show()

#plot_annotations(2)

