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

from EvaluateDataset import Represent, MODELS, count_files_in_directory, DATASET, VALDS, PlotResults, CheckForCheckpoints, Save, THRESHOLD, FOLDER, FILE


ModelData = {
    "Facenet": {"Threshold": 0.66, "Sensibility": 1.1, "MinIds": 7},
    "ArcFace": {"Threshold": 0.45, "Sensibility": 1.3, "MinIds": 5},
    "GhostFaceNet": {"Threshold": 0.5, "Sensibility": 1.1, "MinIds": 7},
    "Facenet512": {"Threshold": 0.68, "Sensibility": 1.05, "MinIds": 2},
    "SFace": {"Threshold": 0.5, "Sensibility": 1.1, "MinIds": 7},
}

def ValidateModel():
    acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for identity in tqdm(os.listdir(VALDS), "evaluating images",position=1, leave=True):
        rep = {}
        repFake = {}
        realRef = os.path.join(VALDS, identity, 'real')
        fakeFolder = os.path.join(VALDS, identity, 'fake')

        for model in ModelData.keys():
            rep[model] = {}
            repFake[model]={}
            for real in os.listdir(realRef):
                rep[model][real] = Represent(os.path.join(realRef, real), model)
            for fake in os.listdir(fakeFolder):
                repFake[model][fake] = Represent(os.path.join(fakeFolder, fake), model)

        #fake evaluation
        for fake in os.listdir(fakeFolder):
            distance = 0    
            votes = 0
            for model in  ModelData.keys():          
                simList = []   
                for real in os.listdir(realRef): 
                    sim = torch.nn.functional.cosine_similarity(rep[model][real], repFake[model][fake], dim=0)
                    distance = max(sim, distance)
                    simList.append(sim)
                 
                if(distance > ModelData[model]["Threshold"] * ModelData[model]["Sensibility"]):
                    votes += 1
                else:
                    #in this case we require a min number of identities to be greater than the threshold
                    acceptables = 0
                    simList.sort(reverse=True)
                    for s in simList:
                        if(s > ModelData[model]["Threshold"]):
                            acceptables += 1
                            if(acceptables >= ModelData[model]["MinIds"]):
                                votes+=1
                                break           
            #check votes
            if(votes >= 2):
                acc['FP'] += 1
            else:
                acc['TN'] += 1

        #pick two random real images and perform evaluation
        toEval = count_files_in_directory(fakeFolder)
        i = 0
        for randomReal in os.listdir(realRef):
            if(i == toEval):
                break
            i += 1
            distance = 0    
            votes = 0
            for model in  ModelData.keys():          
                simList = []   
                for real in os.listdir(realRef): 
                    sim = torch.nn.functional.cosine_similarity(rep[model][real], rep[model][randomReal], dim=0)
                    distance = max(sim, distance)
                    simList.append(sim)
                 
                if(distance > ModelData[model]["Threshold"] * ModelData[model]["Sensibility"]):
                    votes += 1
                else:
                    #in this case we require a min number of identities to be greater than the threshold
                    acceptables = 0
                    simList.sort(reverse=True)
                    for s in simList:
                        if(s > ModelData[model]["Threshold"]):
                            acceptables += 1
                            if(acceptables >= ModelData[model]["MinIds"]):
                                votes+=1
                                break           
            #check votes
            if(votes >= 2):
                acc['TP'] += 1
            else:
                acc['FN'] += 1 

    FP = acc['FP']
    FN = acc['FN']
    TP = acc['TP']     
    TN = acc['TN']

    # F1 Score + Accuracy
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    #generate the confusion matrix
    cm = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Positive', 'Predicted Negative'],
            yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title(f'Confusion Matrix TOP 3')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    print("Model: ", model)
    print("F1 score: ", f1)
    print("Accuracy: ", accuracy)
    input("Press Enter to continue...")
    plt.show()
    return

ValidateModel()