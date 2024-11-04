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
from EvaluateDataset import Represent, MODELS, count_files_in_directory, DATASET, VALDS, PlotResults
from scipy.spatial.distance import euclidean, cityblock, mahalanobis

MIN_NUM_IDENTITES = [2,3,4,5,6,7,8,9,10]

def MinIdValidation(model, threshold):
    acc = {}
    for elem in MIN_NUM_IDENTITES:
        acc[elem] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for identity in tqdm(os.listdir(VALDS), "Validation: ",position=1, leave=True):
        rep = {}
        repFake = {}
        realRef = os.path.join(VALDS, identity, 'real')
        fakeFolder = os.path.join(VALDS, identity, 'fake')

        for real in os.listdir(realRef):
            rep[real] = Represent(os.path.join(realRef, real), model)
        for fake in os.listdir(fakeFolder):
            repFake[fake] = Represent(os.path.join(fakeFolder, fake), model)

        #fake evaluation (for each fake image)
        for fake in repFake.keys():
            distance = 0
            acceptables = 0
            for real in rep.keys(): 
                sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0)
                if(sim > threshold): # if the distance is greater than the threshold the model will label the image as Real
                    acceptables += 1
            
            for elem in MIN_NUM_IDENTITES:
                if(acceptables >= elem):
                    acc[elem]['FP'] += 1
                else:
                    acc[elem]['TN'] += 1       
        
        # Real Evaluation (for each real image)
        toEval = count_files_in_directory(fakeFolder)
        i = 0
        for randomReal in rep.keys():
            if(i >= toEval):
                break
            i += 1
            acceptables = 0
            for real in rep.keys():
                if(real == randomReal):
                    continue
                sim = torch.nn.functional.cosine_similarity(rep[real], rep[randomReal], dim=0)
                
                if(sim > threshold):
                    acceptables += 1
            
            
            for elem in MIN_NUM_IDENTITES:
                if(acceptables >= elem):
                    acc[elem]['TP'] += 1
                else:
                    acc[elem]['FN'] += 1      

    # print Results
    #get the min identities that maximizes f1 score
    mF1 = 0
    bestElem = 0
    for elem in acc.keys():
        FP = acc[elem]['FP']
        FN = acc[elem]['FN']
        TP = acc[elem]['TP']     
        TN = acc[elem]['TN']

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        if(f1 > mF1):
            mF1 = f1
            bestElem = elem
        
    print("The best number of identities is: ", bestElem)
    PlotResults(acc[bestElem]['FP'],acc[bestElem]['FN'] ,acc[bestElem]['TP'], acc[bestElem]['TN'],model, threshold)




MinIdValidation(MODELS[9], 0.5)
