import os.path
import numpy as np
import pandas as pd
import csv


mode = "local" #"riv"
main_folder = os.path.dirname(os.path.abspath(__file__))
set_folder = os.path.join(main_folder, "Rachel")
#anno_folder = os.path.join(main_folder, "labels_training_set_daisy")
filename = "frame_MasterList.csv"

# Fields
fields = ["image_path", "annotation_path"]
excluded = [".DS_Store"]

rows = []
for set in os.listdir(set_folder):
    if set not in excluded:
        temp_row = []

        img_name = set[:-4]

        img_path = os.path.join(set_folder, set)
        print(img_path)

        '''ann_path =  os.path.join(anno_folder, img_name+".txt")

        if os.path.isfile(ann_path):
            pass
        else:
            ann_path = np.NaN
            print(f"No annotation for {set}")'''
        df = pd.read_csv("frame_MasterList.csv")
        selfcsv_df = df
        rivanna_mode = False

        imgs_dir = df.image_path
        # print(len(imgs_dir))

        if mode == "local":
            temp_row.append(img_path)
           # temp_row.append(ann_path)
            rows.append(temp_row)
        else:
            img_path = ".." + img_path.split("/sds_capstone")[1]
            ann_path = ".." + ann_path.split("/sds_capstone")[1]

            temp_row.append(img_path)
           # temp_row.append(ann_path)
            rows.append(temp_row)

# writing to csv file
with open(filename, 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)


