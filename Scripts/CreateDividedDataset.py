import requests
from PIL import Image
import base64
from io import BytesIO
import os, random
import tqdm
import time

KEY = "xjO_36phyS6ZD2SOovLssn9ryGmsBg"
datasetFolder = "GoogleDataset/GgDataset"	
newDatasetFolder = "GoogleDataset/GDatasetSplit"
GenerationCost = 0.0096


def CreateNewDataset():
    if(os.path.exists(newDatasetFolder) == False):
        os.mkdir(newDatasetFolder) 
    for file in tqdm.tqdm(os.listdir(datasetFolder), desc="Processing Folders"):

        dataPath = os.path.join(datasetFolder, file)

        path = os.path.join(newDatasetFolder, file)
        realP = os.path.join(path, "real")
        fakeP = os.path.join(path, "fake")
        os.mkdir(path)
        os.mkdir(realP)
        os.mkdir(fakeP)

        for image in os.listdir(dataPath):
            if(image.endswith(".jpg") == False):
                continue
            imagepath = os.path.join(dataPath, image)
            with Image.open(imagepath).convert('RGB') as img:
                img.save(os.path.join(realP, image))

CreateNewDataset()