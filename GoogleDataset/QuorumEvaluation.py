from deepface import DeepFace
import torch
from tqdm import tqdm
import os, random
import math
from pathlib import Path
from PIL import Image
from numpy import var
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

DATASET = "GoogleDataset/GDatasetSplitFolders/val"
#the models to be used in the quorum evaluation
MODELS = [
  "VGG-Face",

  "Facenet",
  "Facenet512",
  #"OpenFace",
  #"DeepID",
  #"ArcFace",
  "SFace",
  "GhostFaceNet",
]
#dict containing the threshold of each model
THRESHOLD = {
    "VGG-Face": 0.5,
    "Facenet": 0.7,
    "Facenet512": 0.72,
    "OpenFace": 0.5,
    "DeepID": 0.2,
    "ArcFace": 0.58,
    "SFace": 0.58,
    "GhostFaceNet": 0.5,
}

def count_files_in_directory(directory):
    # Create a Path object
    path = Path(directory)
    # Count only files (ignoring subdirectories)
    return sum(1 for f in path.iterdir() if f.is_file())


def Represent(path, model):
    embedding_objs = DeepFace.represent(
    img_path = path,
    model_name= model,
    enforce_detection=False,
    )
    SUT_embeddings = embedding_objs[0]['embedding']
    return torch.FloatTensor(SUT_embeddings)


def QuorumEval(realFolder, fakeFolder):
    fakeCount = count_files_in_directory(fakeFolder)

    #===================================
    #===========real representation=====  
    #===================================
    representations = {} # dict of dicts, given an image it gives all the model representations for that image
    for image in os.listdir(realFolder):
        modelRepList = {}
        for model in MODELS:        
            rep = Represent(os.path.join(realFolder, image), model)
            modelRepList[model] = rep

        representations[image] = modelRepList

    #===================================
    #===========fake representation=====
    #===================================

    fakeRepresentations = {} # dict of dicts, given an image it gives all the model representations for that image
    for image in os.listdir(fakeFolder):
        modelRepList = {}
        for model in MODELS:       
            rep = Represent(os.path.join(fakeFolder, image), model)
            modelRepList[model] = rep

        fakeRepresentations[image] = modelRepList

    #===================================
    #===========quorum evaluation=======
    #===================================
    # checking accuracy for fake images
    acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    for image in os.listdir(fakeFolder):
        votes = []
        for model in MODELS:
            currentRep = fakeRepresentations[image][model]
            distance = 0          
            for realImage in os.listdir(realFolder):
                realRep = representations[realImage][model]
                sim = torch.nn.functional.cosine_similarity(realRep, currentRep, dim=0)
                distance = max(sim, distance)

            if(distance > THRESHOLD[model]):
                votes.append(1)
            else:
                votes.append(0)
        
        #if the majority of the models voted for fake, then the image is fake
        if(votes.count(1) > votes.count(0)):
            acc['FP'] +=1 # the quorum guessed wrongly the image
        else:
            acc['TN'] +=1 # the quorum guessed correctly the image

    # checking accuracy for real images
    for i in range(fakeCount):
        votes = []
        randomImage = random.choice(os.listdir(realFolder))
        for model in MODELS:
            currentRep = representations[randomImage][model]
            distance = 0          
            for realImage in os.listdir(realFolder):
                if(realImage == randomImage):
                    continue
                realRep = representations[realImage][model]
                sim = torch.nn.functional.cosine_similarity(realRep, currentRep, dim=0)
                distance = max(sim, distance)

            if(distance > THRESHOLD[model]):
                votes.append(1)
            else:
                votes.append(0)
        
        #if the majority of the models voted for real, then the image is real
        if(votes.count(1) > votes.count(0)):
            acc['TP'] +=1 # the quorum guessed correctly the image
        else:
            acc['FN'] +=1 # the quorum guessed wrong the image

    return acc

def FullEvaluation():
    acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    for identity in tqdm(os.listdir(DATASET), "evaluating images",position=0, leave=True):

        realRef = os.path.join(DATASET, identity, 'real')
        fakeFolder = os.path.join(DATASET, identity, 'fake')

        _acc = QuorumEval(realRef, fakeFolder)
        acc["TP"] += _acc["TP"] 
        acc["TN"] += _acc["TN"]
        acc["FP"] += _acc["FP"]
        acc["FN"] += _acc["FN"]

               
    FP = acc['FP']
    FN = acc['FN']
    TP = acc['TP']     
    TN = acc['TN']
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    plt.figure(figsize=(12, 5))
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    # Add labels and title
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')

    cm = np.array([[TN, FP], [FN, TP]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Positive', 'Predicted Negative'],
            yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title(f'Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Show the plot
    plt.tight_layout()
    
    #return the threshold having the best accuracy and the threshold having the best F1 score 
    #Print The results
    print("F1 score: ", round(f1, 4))
    print("Accuracy: ", round(accuracy, 4))
    plt.show()

FullEvaluation()