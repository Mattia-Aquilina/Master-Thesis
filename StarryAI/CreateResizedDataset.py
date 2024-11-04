import os
import tqdm
from pathlib import Path
from PIL import Image

DATASET = 'StarryAI/DatasetRealfake'
RESIZED_DATASET_PATH = "StarryAI/ResizedDataset"

def count_files_in_directory(directory):
    # Create a Path object
    path = Path(directory)
    # Count only files (ignoring subdirectories)
    return sum(1 for f in path.iterdir() if f.is_file())

def count_directories_in_directory(directory):
    # Create a Path object
    path = Path(directory)
    # Count only directories (ignoring files)
    return sum(1 for f in path.iterdir() if f.is_dir())

def CreateNewDataset():
    if(os.path.exists(RESIZED_DATASET_PATH) == False):
        os.mkdir(RESIZED_DATASET_PATH) 
    else:
        os.remove(RESIZED_DATASET_PATH)

    for file in tqdm.tqdm(os.listdir(DATASET), desc="Processing Folders"):
        #identity folder (eg: /Akhmed_Zakayev)
        dataPath = os.path.join(DATASET, file)
        path = os.path.join(RESIZED_DATASET_PATH, file)

        #check if, for the current identity, the fake sample exsists
        if(count_files_in_directory(os.path.join(dataPath, "fake")) == 0):
            continue

        #create the folders for the real and fake images
        realP = os.path.join(path, "real")
        fakeP = os.path.join(path, "fake")
        os.mkdir(path)
        os.mkdir(realP)
        os.mkdir(fakeP)

        realSamples = os.path.join(dataPath, "real")
        #copy the real images to the new dataset
        for image in os.listdir(realSamples):
            imagepath = os.path.join(realSamples, image)
            with Image.open(imagepath) as img:
                img.save(os.path.join(realP, image))

        #for fake images we have to resized them to 250x250
        fakeSamples = os.path.join(dataPath, "fake")
        for image in os.listdir(fakeSamples):
            imagepath = os.path.join(fakeSamples, image)
            with Image.open(imagepath) as img:
                img = img.resize((250, 250))
                img.save(os.path.join(fakeP, image))


#CreateNewDataset()
print(count_directories_in_directory(RESIZED_DATASET_PATH))