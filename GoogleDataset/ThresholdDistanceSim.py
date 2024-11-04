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

MIN_ID = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
Ss = ['1.05', '1.07','1.08','1.1' , '1.25','1.15','1.2', '1.25', '1.3' , '1.35']


def FullEvaluation(model):
    ckp = CheckForCheckpoints()
    acc = {}
    LastIdentity = ""
    if(ckp):
        acc = ckp['acc']
        LastIdentity = ckp['id']
    else:
        #create checkpoint file

        for t in THRESHOLD:
            acc[t] = {}
            for sim in Ss:
                    acc[t][sim] = {}
                    for elem in MIN_ID:
                        acc[t][sim][elem] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        LastIdentity = ""

    for identity in tqdm(os.listdir(DATASET), "evaluating images",position=0, leave=True):
        if(identity <= LastIdentity):
            continue
             
        rep = {}
        repFake = {}
        realRef = os.path.join(DATASET, identity, 'real')
        fakeFolder = os.path.join(DATASET, identity, 'fake')

        for real in os.listdir(realRef):
            rep[real] = Represent(os.path.join(realRef, real), model)      
        for fake in os.listdir(fakeFolder):
            repFake[fake] = Represent(os.path.join(fakeFolder, fake), model)
            

        #fake evaluation
        for fake in repFake.keys():
            distance = 0
            simList = []
            for real in rep.keys(): 
                sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0)
                simList.append(sim)
                distance = max(sim, distance)
            simList.sort(reverse=True)

            for threshold in THRESHOLD:
                for sim in Ss:
                    for elem in MIN_ID:
                        if(distance > float(threshold) * float(sim)):
                            acc[threshold][sim][elem]['FP'] += 1
                        else:
                            #in this case we require a min number of identities to be greater than the threshold
                            acceptables = 0
                            for s in simList:
                                if(s > float(threshold)):
                                    acceptables += 1
                                    if(acceptables >= int(elem)):
                                        acc[threshold][sim][elem]['FP'] += 1
                                        break
                            if(acceptables < int(elem)):
                                acc[threshold][sim][elem]['TN'] += 1

        #pick random real images and perform evaluation
        toEval = count_files_in_directory(fakeFolder)
        i = 0
        for randomReal in rep.keys():
            if(i == toEval):
                break
            i += 1
            distance = 0
            simList = []
            for real in rep.keys():
                if(real == randomReal):
                    continue
                sim = torch.nn.functional.cosine_similarity(rep[randomReal], rep[real], dim=0)
                distance = max(sim, distance)
                simList.append(sim)
            simList.sort(reverse=True)

            for threshold in THRESHOLD:
                for sim in Ss:
                    for elem in MIN_ID:
                        if(distance > float(threshold) * float(sim)):
                            acc[threshold][sim][elem]['TP'] += 1
                        else:
                            #in this case we require a min number of identities to be greater than the threshold
                            acceptables = 0
                            for s in simList:
                                if(s > float(threshold)):
                                    acceptables += 1
                                    if(acceptables >= int(elem)):
                                        acc[threshold][sim][elem]['TP'] += 1    
                                        break
                            if(acceptables < int(elem)):
                                acc[threshold][sim][elem]['FN'] += 1
        #save the checkpoint
        Save(acc, identity)      

    #remove checkpoints
    os.remove(os.path.join(FOLDER, FILE)) 
    Results(acc, model)

def Results(acc, model):
    #find the best values of threshold, min identities and similarity

    best = {'threshold': 0, 'sim': 0, 'elem': 0, 'f1': 0, 'accuracy': 0}
    bestData = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for threshold in THRESHOLD:
        for sim in Ss:
            for elem in MIN_ID:
                TP = acc[threshold][sim][elem]['TP']
                TN = acc[threshold][sim][elem]['TN']
                FP = acc[threshold][sim][elem]['FP']
                FN = acc[threshold][sim][elem]['FN']
                #compute f1 score
                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

                if(f1 > best['f1']):
                    bestData['TP'] = TP
                    bestData['TN'] = TN
                    bestData['FP'] = FP
                    bestData['FN'] = FN

                    best['f1'] = f1
                    best['threshold'] = threshold
                    best['sim'] = sim
                    best['elem'] = elem
                    best['accuracy'] = (TP + TN) / (TP + TN + FP + FN)

    # compute the best data for a given threshold so to plot the ROC curve
    tpr_list = []
    fpr_list = []
    for threshold in THRESHOLD:
        _best = {'f1':0, 'TP':0, 'TN':0, 'FP':0, 'FN':0}
        for sim in Ss:
            for elem in MIN_ID:
                TP = acc[threshold][sim][elem]['TP']
                TN = acc[threshold][sim][elem]['TN']
                FP = acc[threshold][sim][elem]['FP']
                FN = acc[threshold][sim][elem]['FN']
                #compute f1 score
                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

                if(f1 > _best['f1']):
                    _best['f1'] = f1
                    _best['TP'] = TP
                    _best['TN'] = TN
                    _best['FP'] = FP
                    _best['FN'] = FN
        
        tpr = _best['TP'] / (_best['TP'] + _best['FN']) if (_best['TP'] + _best['FN']) != 0 else 0
        fpr = _best['FP'] / (_best['FP'] + _best['TN']) if (_best['FP'] + _best['TN']) != 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    #roc curve
    #roc_auc = auc(fpr_list, tpr_list)
    # Plot the ROC curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_list, tpr_list, marker='o', linestyle='-', color='b')
# Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # Add labels and title
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')

    #plot the confusion matrix
    bestTp = bestData['TP']
    bestTn = bestData['TN']
    bestFp = bestData['FP']
    bestFn = bestData['FN']

    cm = np.array([[bestTn, bestFp], [bestFn, bestTp]])
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Positive', 'Predicted Negative'],
            yticklabels=['Actual Positive', 'Actual Negative'])
    bf1 = best['f1']
    belem = best['elem']
    sens = best['sim']
    plt.title(f'Confusion Matrix (Threshold = {bf1}, Min_Ids = {belem}, Sens = {sens})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Show the plot
    plt.tight_layout()

    #print("AUC", roc_auc)
    print("The best threshold is: ", best['threshold'])
    print("The best similarity sensitivity is: ", best['sim'])
    print("The best number of identities is: ", best['elem'])
    print("f1 score is: ", best['f1'])
    print("accuracy is: ", best['accuracy'])
    input("Press Enter to continue...")
    plt.show()


    return (float(best['threshold']), float(best['sim']), int(best['elem']))

def ValidateModel(model, threshold, sens, minIds):
    acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for identity in tqdm(os.listdir(VALDS), "evaluating images",position=1, leave=True):
        rep = {}
        repFake = {}
        realRef = os.path.join(VALDS, identity, 'real')
        fakeFolder = os.path.join(VALDS, identity, 'fake')

        for real in os.listdir(realRef):
            rep[real] = Represent(os.path.join(realRef, real), model)
        for fake in os.listdir(fakeFolder):
            repFake[fake] = Represent(os.path.join(fakeFolder, fake), model)

        #fake evaluation
        for fake in repFake.keys():
            distance = 0
            simList = []
            for real in rep.keys(): 
                sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0)
                distance = max(sim, distance)
                simList.append(sim)
            
            if(distance > float(threshold) * float(sens)):
                acc['FP'] += 1
            else:
                #in this case we require a min number of identities to be greater than the threshold
                acceptables = 0
                simList.sort(reverse=True)
                for s in simList:
                    if(s > float(threshold)):
                        acceptables += 1
                        if(acceptables >= int(minIds)):
                            acc['FP'] += 1
                            break
                if(acceptables < int(minIds)):
                    acc['TN'] += 1 

        #pick two random real images and perform evaluation
        toEval = count_files_in_directory(fakeFolder)
        i = 0
        for randomReal in rep.keys():
            if(i == toEval):
                break
            i += 1
            distance = 0
            simList = []
            for real in rep.keys():
                if(real == randomReal):
                    continue
                sim = torch.nn.functional.cosine_similarity(rep[real], rep[randomReal], dim=0)
                distance = max(sim, distance)
                simList.append(sim)
            
            if(distance > float(threshold) * float(sens)):
                acc['TP'] += 1
            else:
                simList.sort(reverse=True)
                for s in simList:
                    if(s > float(threshold)):
                        acceptables += 1
                        if(acceptables >= int(minIds)):
                            acc['TP'] += 1
                            break
                if(acceptables < int(minIds)):
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
    plt.title(f'Confusion Matrix (Threshold = {threshold})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    print("Model: ", model)
    print("F1 score: ", f1)
    print("Accuracy: ", accuracy)
    input("Press Enter to continue...")
    plt.show()
    return

model = MODELS[6] # cambiare in 8
#print("Model: ", model)
#FullEvaluation(model)
# print(sens)
# exit()
ValidateModel(model, 0.45 , 1.3 , 5)

