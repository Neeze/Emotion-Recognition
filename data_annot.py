import os
import csv
import random
import gdown
import gdown
import zipfile

url = "https://drive.google.com/file/d/1lbvrFHux0XHr7Ka-ayKHBOONYnxKctrS/view?usp=drive_link"
output = "archive.zip"
gdown.download(url=url, output=output, fuzzy=True)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data')


data_dir = "data"

labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def create_csv(data_type, labels):
    csv_file_path = os.path.join(data_dir, f"{data_type}.csv")
    data_rows = []

    for label in labels:
        label_dir = os.path.join(data_dir, data_type, label)
        for filename in os.listdir(label_dir):
            if filename.endswith(".jpg"):
                image_path = os.path.join(label_dir, filename)
                data_rows.append([image_path, label])

    random.shuffle(data_rows)

    with open(csv_file_path, "w", newline="") as file: 
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])
        writer.writerows(data_rows)

create_csv("train", labels)
create_csv("test", labels)

print("Done!")

