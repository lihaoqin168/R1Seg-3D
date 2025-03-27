import os
from tqdm import tqdm
import re
import csv
import json


root_path = './Data/data/M3D_RefSeg_npy/'

path = "./Data/data/RefSegData/RefSeg_data.csv"
with open(path, "a", newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ["Image", "Mask", "Mask_ID", "Text", "Question_Type", "Question", "Answer"])

import pandas as pd
import json

# 读取CSV文件
df = pd.read_csv('/M3D-RefSeg/M3D_RefSeg_test.csv')

# 遍历DataFrame中的每一行
for index, row in df.iterrows():
    txt_path = row['Image'].replace('ct.nii.gz','text.json')
    img_path = row['Image'].replace('.nii.gz','.npy')
    mask_path = row['Mask'].replace('.nii.gz','.npy')
    Mask_ID = row['Mask_ID']
    question_type = row['Question_Type']
    question = row['Question']
    answer = row['Answer']

    txt_path = os.path.join(root_path, txt_path)
    img_path = os.path.join(root_path, img_path)
    mask_path = os.path.join(root_path, mask_path)

    try:
        # 读取JSON文件
        with open(txt_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text = data[Mask_ID]

        with open(path, "a", newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([img_path, mask_path, Mask_ID, text, question_type, question, answer])

    except FileNotFoundError:
        print(f"File not found: {path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {path}")





