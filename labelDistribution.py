import os.path
import numpy as np
import csv


mode = "local" #"riv"
main_folder = os.path.dirname(os.path.abspath(__file__))
set_folder = os.path.join(main_folder, "train_set_daisy")
anno_folder = os.path.join(main_folder, "relabel_1")
recode_folder = os.path.join(main_folder, "recoded_labels_crop")
filename = "frame_MasterList.csv"

# Fields
fields = ["image_path", "annotation_path"]
excluded = [".DS_Store"]

rows = []
for file in os.listdir(anno_folder):
    if file not in excluded:
        ann_path = os.path.join(anno_folder, file)
        print(ann_path)
        save_path = os.path.join(recode_folder, file)

        with open(file) as f:
            lines = [line.rstrip('\n') for line in f]
        num_objs = len(lines)

        list_anns = []
        for i in range(num_objs):
            current = lines[i]
            contents = current.split(" ")
            if len(contents) != 5:
                print("Error: Annotation does not contain 4 coordinates or is mising label")
            else:
                class_label = int(contents[0])
                class_label = 1
                other = ''.join(contents[1:])
                concat_string = str(class_label) + " " + str(other)
                list_anns.append(concat_string)

        with open(save_path, mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(list_anns))

        print(len(list_anns))