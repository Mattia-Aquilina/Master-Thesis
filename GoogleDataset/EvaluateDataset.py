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
from transformers import CLIPProcessor, CLIPModel
import json

fw = 0.9
cw = 0.1
#CHECKPOINTS
FOLDER = "Checkpoints"
FILE = "ckpt.json"
#DATA
THRESHOLD = ['0.2', '0.22', '0.24', '0.26', '0.28', '0.3', '0.32', '0.34', '0.36', '0.38', '0.4', '0.45', '0.5', '0.52', '0.54', '0.56', '0.58', '0.6', '0.62', '0.64', '0.66', '0.68', '0.7', '0.72', '0.74', '0.76', '0.78', '0.8', '0.82', '0.84', '0.86', '0.88', '0.9', '0.92', '0.94', '0.96', '0.98', '1']
#THRESHOLD = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5 ,16, 16.5, 17, 17.5 , 18, 18.5, 19, 19.5, 20, 20.5, 21] #from 1 to 30
DATASET = "GoogleDataset/GDatasetSplitFolders/train"
VALDS = "GoogleDataset/GDatasetSplitFolders/val"
MODELS = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]


def Represent(path, model):  
    embedding_objs = DeepFace.represent(
    img_path = path,
    model_name= model,
    enforce_detection=False,
    )
    SUT_embeddings = embedding_objs[0]['embedding']
    return torch.FloatTensor(SUT_embeddings)

def ClipRepresent(path):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(path).convert("RGB")
    input = processor(images=image, return_tensors="pt")
    f =  model.get_image_features(**input)
    return f / f.norm(dim=-1, keepdim=True)
     
                             
def count_files_in_directory(directory):
    # Create a Path object
    path = Path(directory)
    # Count only files (ignoring subdirectories)
    return sum(1 for f in path.iterdir() if f.is_file())

def CheckForCheckpoints():
    if os.path.exists(os.path.join(FOLDER, FILE)):
        with open(os.path.join(FOLDER, FILE), 'r') as f:
            data = json.load(f)
        return data
    else:
        os.makedirs(FOLDER, exist_ok=True)
        with open(os.path.join(FOLDER, FILE), 'w') as f:
            json.dump({}, f)
        return False
        
def Save(acc, identity):
    data = {}
    data['id'] = identity
    data['acc'] = acc
    
    with open(os.path.join(FOLDER, FILE), 'w') as f:
        json.dump(data, f)

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
            acc[t] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        LastIdentity = ""

    for identity in tqdm(os.listdir(DATASET), "evaluating images",position=1, leave=True):
        if(identity <= LastIdentity):
            continue
             
        rep = {}
        repFake = {}
        clipRep = {}
        clipRepFake = {}
        realRef = os.path.join(DATASET, identity, 'real')
        fakeFolder = os.path.join(DATASET, identity, 'fake')

        for real in os.listdir(realRef):
            rep[real] = Represent(os.path.join(realRef, real), model)      
            clipRep[real] = ClipRepresent(os.path.join(realRef, real)) 
        for fake in os.listdir(fakeFolder):
            repFake[fake] = Represent(os.path.join(fakeFolder, fake), model)
            clipRepFake[fake] = ClipRepresent(os.path.join(fakeFolder, fake))
            

        #fake evaluation
        for fake in repFake.keys():
            distance = 0
            for real in rep.keys(): 
                sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0) * fw + torch.nn.functional.cosine_similarity(clipRep[real], clipRepFake[fake], dim=-1).item() * cw
                distance = max(sim, distance)

            for threshold in THRESHOLD:
                if(distance > float(threshold)):
                    acc[threshold]['FP'] += 1
                else:
                    acc[threshold]['TN'] += 1 

        #pick random real images and perform evaluation
        toEval = count_files_in_directory(fakeFolder)
        i = 0
        for randomReal in rep.keys():
            if(i == toEval):
                break
            i += 1
            distance = 0
            for real in rep.keys():
                if(real == randomReal):
                    continue
                sim = torch.nn.functional.cosine_similarity(rep[randomReal], rep[real], dim=0)*fw + torch.nn.functional.cosine_similarity(clipRep[randomReal], clipRep[real], dim=-1).item() * cw
                distance = max(sim, distance)
            
            for threshold in THRESHOLD:
                if(distance > float(threshold)):
                    acc[threshold]['TP'] += 1
                else:
                    acc[threshold]['FN'] += 1        

        #save the checkpoint

        Save(acc, identity)      

    #remove checkpoints
    os.remove(os.path.join(FOLDER, FILE)) 
    return acc


def EvaluateModel(model):
    #store the accuracy in the form Threshold : {TP, TN, FP, FN}
    result = FullEvaluation(model)
    
    #Extract the data from the evaluation results
    f1 = {}
    acc = {}
    tpr_list = []
    fpr_list = []
    threshold_list = []
    for t in THRESHOLD:
        FP = result[t]['FP']
        FN = result[t]['FN']
        TP = result[t]['TP']     
        TN = result[t]['TN']

        # F1 Score + Accuracy
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        
        f1[t] = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        acc[t] = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        #ROC Curve
        tpr = TP / (TP + FN) if (TP + FN) != 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) != 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        threshold_list.append(t)


    tf1 = max(f1, key=f1.get)

    #plot the ROC curve
    roc_auc = auc(fpr_list, tpr_list)
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
    FP = result[tf1]['FP']
    FN = result[tf1]['FN']
    TP = result[tf1]['TP']     
    TN = result[tf1]['TN']

    cm = np.array([[TN, FP], [FN, TP]])
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Positive', 'Predicted Negative'],
            yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title(f'Confusion Matrix (Threshold = {tf1})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Show the plot
    plt.tight_layout()
    
    #return the threshold having the best accuracy and the threshold having the best F1 score   
    #Print The results
    print("Model: ", model)
    print("Threshold:", round(float(tf1), 4))
    print("F1 score: ", round(f1[tf1], 4))
    print("Accuracy: ", round(acc[tf1], 4))
    #print("The treshold maximizing Accuracy is ", tacc, " with F1 score of ", f1[tacc], " and Accuracy of ", acc[tacc])
    print("The AUC is ", round(roc_auc, 4))
    input("Press Enter to continue...")
    plt.show()
    

def PlotResults(FP, FN, TP, TN, model, threshold):
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


def EvaluateModels():
    _ts = {}
    for model in tqdm(MODELS, desc="processing models", position=0, leave=True):
        result = EvaluateModel(model) #function that finds the best threshold for a given model
        _ts[model] = result
        
    print(_ts)

def ValidateModel(model, threshold):
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
            for real in rep.keys(): 
                sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0)
                distance = max(sim, distance)

            
            if(distance > threshold):
                acc['FP'] += 1
            else:
                acc['TN'] += 1 

        #pick two random real images and perform evaluation
        toEval = count_files_in_directory(fakeFolder)
        i = 0
        for randomReal in rep.keys():
            if(i == toEval):
                break
            i += 1
            distance = 0
            for real in rep.keys():
                if(real == randomReal):
                    continue
                sim = torch.nn.functional.cosine_similarity(rep[real], rep[randomReal], dim=0)
                distance = max(sim, distance)
            
            
            if(distance > threshold):
                acc['TP'] += 1
            else:
                acc['FN'] += 1       

    FP = acc['FP']
    FN = acc['FN']
    TP =acc['TP']     
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
          
    

# M = MODELS[9]
# res = EvaluateModel(M)
#ValidateModel(M,0.5)
