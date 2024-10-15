import csv
from tqdm import tqdm
import os
from PIL import Image
import base64
from io import BytesIO

DATASET = "GDatasetSplit"
def Base64EncodeImage(image):
    with Image.open(image) as img:
        buffered = BytesIO()
        img = img.convert("RGB")
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    

def CreateCSV():
    with open('gdataset.csv', mode='w') as names_file:
        names_writer = csv.writer(names_file, delimiter=',')

        for identity in tqdm(os.listdir(DATASET), "evaluating identity",position=0, leave=True):
            
            for file in tqdm(os.listdir(os.path.join(DATASET, identity,"real")) , "processing reals",position=1, leave=False):
                names_writer.writerow( [identity, "real", Base64EncodeImage(os.path.join(DATASET, identity, "real", file))])
            for file in tqdm(os.listdir(os.path.join(DATASET, identity,"fake")) , "processing fakes",position=1, leave=False):
                names_writer.writerow( [identity, "fake", Base64EncodeImage(os.path.join(DATASET, identity, "fake", file))])
        
# def FromCSV(path, savePath):


CreateCSV()