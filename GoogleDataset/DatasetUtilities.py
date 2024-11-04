from pathlib import Path
import os
import collections
import splitfolders
import random
import shutil
from tqdm import tqdm
def count_files_in_split_dir(directory):
    # Create a Path object
    fakes = sum(1 for fake in Path(os.path.join(directory, 'fake')).iterdir() if fake.is_file())
    reals = sum(1 for real in Path(os.path.join(directory, 'real')).iterdir() if real.is_file())

    return (fakes, reals)


def count_files_in_directory(directory):
    # Create a Path object
    path = Path(directory)
    # Count only files (ignoring subdirectories)
    return sum(1 for f in path.iterdir() if f.is_file())

# Replace "directory_path" with the path to your specific folder


def CountDatasetEntry(directory_path):
    identities=  sum(1 for f in Path(directory_path).iterdir() if f.is_dir())
    print(f"identities : " +str(identities))
    count = 0
    realUnder25 = 0 
    fakeToGen = 0
    for file in os.listdir(directory_path):
        path = os.path.join(directory_path, file)
        if os.path.isdir(path):
            fake, real = count_files_in_split_dir(path)
            fakeToGen += (15 - fake)
            if(real < 25):
                print(f"Identity {file} has less than 25 real images")
                realUnder25 += 1
        
    print(f"Total fake images to generate : {fakeToGen}")
    print(f"Total cost of generating fake images : {fakeToGen * 0.0096}")
    print(f"Total identities with less than 25 real images : {realUnder25}")

def SplitFolders(dataset):
    newDs = dataset + "Folders"

    trainDs = os.path.join(newDs, "train")
    valDs = os.path.join(newDs, "val")
    os.makedirs(newDs, exist_ok=True)
    os.makedirs(trainDs, exist_ok=True)
    os.makedirs(valDs, exist_ok=True)

    for folder in tqdm(os.listdir(dataset), desc="Splitting folders", position=0, leave=True):

        os.makedirs(os.path.join(trainDs, folder), exist_ok=True)
        os.makedirs(os.path.join(valDs, folder), exist_ok=True)
        os.makedirs(os.path.join(trainDs, folder, "fake"), exist_ok=True)
        os.makedirs(os.path.join(trainDs, folder, "real"), exist_ok=True)
        os.makedirs(os.path.join(valDs, folder, "fake"), exist_ok=True)
        os.makedirs(os.path.join(valDs, folder, "real"), exist_ok=True)

       
        reals = os.path.join(dataset, folder, "real")
        reals= os.listdir(reals)
        random.shuffle(reals)
        i = 0
        for image in reals:
            img = os.path.join(dataset, folder, "real", image)
            if(i < 15):
                shutil.copy(img, os.path.join(trainDs, folder, "real"))
            else:
                shutil.copy(img, os.path.join(valDs, folder, "real"))
            i += 1
        fake = os.path.join(dataset, folder, "fake")
        fake = os.listdir(fake)
        random.shuffle(fake)
        i = 0
        for image in fake:
            img = os.path.join(dataset, folder, "fake", image)
            if(i < 10):
                shutil.copy(img, os.path.join(trainDs, folder, "fake"))
            else:
                shutil.copy(img, os.path.join(valDs, folder, "fake"))
            i += 1

dataset = "GoogleDataset/GDatasetSplit"
SplitFolders(dataset)