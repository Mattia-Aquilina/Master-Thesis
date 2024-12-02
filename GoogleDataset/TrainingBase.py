from deepface import DeepFace
import torch
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
import itertools
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
##implementazione della pipeline di base. Viene calcolato il threshold che massimizza l'f1 score
class TFRecognition:
    #Checkpoints
    FOLDER = "Checkpoints"
    FILE = "ckpt.json"
    #generic data
    model = ""
    TrainDs =""
    ValDs = ""
    #constructor
    def __init__(self, model, TrainDs, ValDs):
        self.model = model
        self.TrainDs = TrainDs
        self.ValDs = ValDs



    def Similarity(self, rep1, rep2):
        return torch.nn.functional.cosine_similarity(rep1, rep2, dim=0)
    #utilities
    def count_files_in_directory(self, directory):
        # Create a Path object
        path = Path(directory)
        # Count only files (ignoring subdirectories)
        return sum(1 for f in path.iterdir() if f.is_file())

    def Represent(self, path):
        embedding_objs = DeepFace.represent(
        img_path = path,
        model_name= self.model,
        enforce_detection=False,
        )
        SUT_embeddings = embedding_objs[0]['embedding']
        return torch.FloatTensor(SUT_embeddings)
    
    def CheckForCheckpoints(self):
        if os.path.exists(os.path.join(self.FOLDER, self.FILE)):
            with open(os.path.join(self.FOLDER, self.FILE), 'r') as f:
                data = json.load(f)
            return data
        else:
            os.makedirs(self.FOLDER, exist_ok=True)
            with open(os.path.join(self.FOLDER, self.FILE), 'w') as f:
                json.dump({}, f)
            return False
        
    def Save(self, acc, identity):
        data = {}
        data['id'] = identity
        data['acc'] = acc
        
        with open(os.path.join(self.FOLDER, self.FILE), 'w') as f:
            json.dump(data, f)

    #train the model

    def ProcessIdentity(self, realRef, fakeFolder, acc, THRESHOLD):
        rep = {}
        repFake = {}

        for real in os.listdir(realRef):
            rep[real] = self.Represent(os.path.join(realRef, real))      
        for fake in os.listdir(fakeFolder):
            repFake[fake] = self.Represent(os.path.join(fakeFolder, fake))
            
        #fake evaluation
        for fake in repFake.keys():
            distance = 0
            for real in rep.keys(): 
                sim = self.Similarity(rep[real], repFake[fake]) 
                distance = max(sim, distance)

            for threshold in THRESHOLD:
                if(distance > float(threshold)):
                    acc[threshold]['FP'] += 1
                else:
                    acc[threshold]['TN'] += 1 

        #pick random real images and perform evaluation
        toEval = self.count_files_in_directory(fakeFolder)
        i = 0
        for randomReal in rep.keys():
            if(i == toEval):
                break
            i += 1
            distance = 0
            for real in rep.keys():
                if(real == randomReal):
                    continue
                sim = torch.nn.functional.cosine_similarity(rep[randomReal], rep[real], dim=0)
                distance = max(sim, distance)
            
            for threshold in THRESHOLD:
                if(distance > float(threshold)):
                    acc[threshold]['TP'] += 1
                else:
                    acc[threshold]['FN'] += 1     



    def PlotResults(self, acc, THRESHOLD):
        #Extract the data from the evaluation results
        f1 = {}
        acc = {}
        tpr_list = []
        fpr_list = []
        threshold_list = []
        for t in THRESHOLD:
            FP = acc[t]['FP']
            FN = acc[t]['FN']
            TP = acc[t]['TP']     
            TN = acc[t]['TN']

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
        FP = acc[tf1]['FP']
        FN = acc[tf1]['FN']
        TP = acc[tf1]['TP']     
        TN = acc[tf1]['TN']

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
        print("Model: ", self.model)
        print("Threshold:", round(float(tf1), 4))
        print("F1 score: ", round(f1[tf1], 4))
        print("Accuracy: ", round(acc[tf1], 4))
        #print("The treshold maximizing Accuracy is ", tacc, " with F1 score of ", f1[tacc], " and Accuracy of ", acc[tacc])
        print("The AUC is ", round(roc_auc, 4))
        input("Press Enter to continue...")
        plt.show()


    def train(self, Thresholds):
        if(self.TrainDs == "" or self.ValDs == ""):
            print("No dataset provided")
            return
        elif(self.model == ""):
            print("No model provided")
            return
        if(not isinstance(Thresholds, list)):
            print("Thresholds should be a list")
            return
        
        #estimate the thresholds
        ckp = self.CheckForCheckpoints()
        acc = {}
        LastIdentity = ""
        if(ckp):
            print("test")
            acc = ckp['acc']
            LastIdentity = ckp['id']
        else:
            #create checkpoint file
            for t in Thresholds:
                acc[t] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            LastIdentity = ""

        for identity in tqdm(os.listdir(self.TrainDs), "evaluating images",position=0, leave=True):
            if(identity <= LastIdentity):
                continue
            
            realRef = os.path.join(self.TrainDs, identity, 'real')
            fakeFolder = os.path.join(self.TrainDs, identity, 'fake')
            acc =  self.ProcessIdentity(realRef, fakeFolder, acc, Thresholds)
            #save the checkpoint

            self.Save(acc, identity)      

        #remove checkpoints
        os.remove(os.path.join(self.FOLDER, self.FILE)) 
        self.PlotResults(acc, Thresholds)



##implementazione della pipeline Clip + Face recognition
class TFClipRecognition(TFRecognition):
    FaceWeights = []

    def __init__(self, model, TrainDs, ValDs,FaceWeight):
        super().__init__(model, TrainDs, ValDs)
        self.FaceWeights = FaceWeight
    

    def ClipRepresent(self, path):
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        image = Image.open(path).convert("RGB")
        input = processor(images=image, return_tensors="pt")
        f =  model.get_image_features(**input)
        return f / f.norm(dim=-1, keepdim=True)
    
    def ProcessIdentity(self, realRef, fakeFolder, acc, THRESHOLD):
        rep = {}
        repFake = {}
        clipRep = {}
        clipRepFake = {}

        for real in os.listdir(realRef):
            rep[real] = self.Represent(os.path.join(realRef, real))
            clipRep[real] = self.ClipRepresent(os.path.join(realRef, real))      
        for fake in os.listdir(fakeFolder):
            repFake[fake] = self.Represent(os.path.join(fakeFolder, fake))
            clipRepFake[fake] = self.ClipRepresent(os.path.join(fakeFolder, fake))
            
        #fake evaluation
        for weight in self.FaceWeights:
            for fake in repFake.keys():
                distance = 0
                for real in rep.keys(): 
                    sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[real], clipRepFake[fake], dim=1).item() * (1 - float(weight))
                    distance = max(sim, distance)

                for threshold in THRESHOLD:
                    if(distance > float(threshold)):
                        acc[threshold][weight]['FP'] += 1
                    else:
                        acc[threshold][weight]['TN'] += 1 

            #pick random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
            i = 0
            for randomReal in rep.keys():
                if(i == toEval):
                    break
                i += 1
                distance = 0
                for real in rep.keys():
                    if(real == randomReal):
                        continue
                    sim = torch.nn.functional.cosine_similarity(rep[randomReal], rep[real], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[randomReal], clipRep[real], dim=1).item() * (1 - float(weight))
                    distance = max(sim, distance)
                
                for threshold in THRESHOLD:
                    if(distance > float(threshold)):
                        acc[threshold][weight]['TP'] += 1
                    else:
                        acc[threshold][weight]['FN'] += 1  

        return acc
                    
    def train(self, Thresholds):
        if(self.TrainDs == "" or self.ValDs == ""):
            print("No dataset provided")
            return
        elif(self.model == ""):
            print("No model provided")
            return
        if(not isinstance(Thresholds, list)):
            print("Thresholds should be a list")
            return
        
        #estimate the thresholds
        ckp = self.CheckForCheckpoints()
        acc = {}
        LastIdentity = ""
        if(ckp):
            acc = ckp['acc']
            LastIdentity = ckp['id']
        else:
            #create checkpoint file
            for t in Thresholds:
                acc[t] = {}
                for w in self.FaceWeights:
                    acc[t][w] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            LastIdentity = ""

        for identity in tqdm(os.listdir(self.TrainDs), "evaluating images",position=0, leave=True):
            if(identity <= LastIdentity):
                continue
            
            realRef = os.path.join(self.TrainDs, identity, 'real')
            fakeFolder = os.path.join(self.TrainDs, identity, 'fake')
            acc =  self.ProcessIdentity(realRef, fakeFolder, acc, Thresholds)
            #save the checkpoint

            self.Save(acc, identity)      

        #remove checkpoints
        #os.remove(os.path.join(self.FOLDER, self.FILE)) 
        self.PlotResults(acc, Thresholds)

    def PlotResults(self, acc, THRESHOLD):
        #Extract the data from the evaluation results
        tpr_list = []
        fpr_list = []
        threshold_list = []

        best = {'threshold': 0, 'weight': 0, 'f1': 0, 'accuracy': 0}
        bestData = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
 
        for t in THRESHOLD:
            for w in self.FaceWeights:
                FP = acc[t][w]['FP']
                FN = acc[t][w]['FN']
                TP = acc[t][w]['TP']     
                TN = acc[t][w]['TN']

                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0


                if(f1 > best['f1']):
                    bestData['TP'] = TP
                    bestData['TN'] = TN
                    bestData['FP'] = FP
                    bestData['FN'] = FN
                    best['threshold'] = t
                    best['weight'] = w
                    best['f1'] = f1
                    best['accuracy'] = (TP + TN) / (TP + TN + FP + FN)


        # Plot the diagonal line (random classifier)
        plt.figure(figsize=(12, 5))

        # Add labels and title
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        
        #plot the confusion matrix
        tf1 = best['f1']
        acc = best['accuracy']
        w = best['weight']
        t = best['threshold']

        FP = bestData['FP']
        FN = bestData['FN']
        TP = bestData['TP']     
        TN = bestData['TN']

        cm = np.array([[TN, FP], [FN, TP]])
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
        print("Model: ", self.model)
        print("Threshold:", round(float(t), 4))
        print("F1 score: ", round(tf1, 4))
        print("Accuracy: ", round(acc, 4))
        print("Weight: ", round(float(w), 4))
        #print("The treshold maximizing Accuracy is ", tacc, " with F1 score of ", f1[tacc], " and Accuracy of ", acc[tacc])
        input("Press Enter to continue...")
        plt.show()



class TFSensRecognition(TFRecognition):
    Sensibility = []
    MinIDs = []

    def __init__(self, model, TrainDs, ValDs, Sensibility, MinIDs):
        super().__init__(model, TrainDs, ValDs)
        self.Sensibility = Sensibility
        self.MinIDs = MinIDs

    def train(self, Thresholds):
        ckp = self.CheckForCheckpoints()
        acc = {}
        LastIdentity = ""
        if(ckp):
            acc = ckp['acc']
            LastIdentity = ckp['id']
        else:
            #create checkpoint file

            for t in Thresholds:
                acc[t] = {}
                for sim in self.Sensibility:
                        acc[t][sim] = {}
                        for elem in self.MinIDs:
                            acc[t][sim][elem] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            LastIdentity = ""

        for identity in tqdm(os.listdir(self.TrainDs), "evaluating images",position=0, leave=True):
            if(identity <= LastIdentity):
                continue

            realRef = os.path.join(self.TrainDs, identity, 'real')
            fakeFolder = os.path.join(self.TrainDs, identity, 'fake')
            acc =  self.ProcessIdentity(realRef, fakeFolder, acc, Thresholds)
            #save the checkpoint

            self.Save(acc, identity)      

        #remove checkpoints
        os.remove(os.path.join(self.FOLDER, self.FILE)) 
        self.PlotResults(acc, Thresholds)



    def ProcessIdentity(self, realRef, fakeFolder, acc, THRESHOLD):
        rep = {}
        repFake = {}

        for real in os.listdir(realRef):
            rep[real] = self.Represent(os.path.join(realRef, real))      
        for fake in os.listdir(fakeFolder):
            repFake[fake] = self.Represent(os.path.join(fakeFolder, fake))
            

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
                for sim in self.Sensibility:
                    for elem in self.MinIDs:
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
        toEval = self.count_files_in_directory(fakeFolder)
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
                for sim in self.Sensibility:
                    for elem in self.MinIDs:
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

        return acc
    
    def PlotResults(self, acc, THRESHOLD):
        best = {'threshold': 0, 'sim': 0, 'elem': 0, 'f1': 0, 'accuracy': 0}
        bestData = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for threshold in THRESHOLD:
            for sim in self.Sensibility:
                for elem in self.MinIDs:
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
            for sim in self.Sensibility:
                for elem in self.MinIDs:
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
        
        

class TFSensClipRecognition(TFRecognition):
    FaceWeights = []

    def __init__(self, model, TrainDs, ValDs,FaceWeight, Sensibility, MinIDs):
        super().__init__(model, TrainDs, ValDs)
        self.FaceWeights = FaceWeight
        self.Sensibility = Sensibility
        self.MinIDs = MinIDs


    def ClipRepresent(self, path):
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        image = Image.open(path).convert("RGB")
        input = processor(images=image, return_tensors="pt")
        f =  model.get_image_features(**input)
        return f / f.norm(dim=-1, keepdim=True)

    
    def train(self, Thresholds):
        if(self.TrainDs == "" or self.ValDs == ""):
            print("No dataset provided")
            return
        elif(self.model == ""):
            print("No model provided")
            return
        if(not isinstance(Thresholds, list)):
            print("Thresholds should be a list")
            return
        
        #estimate the thresholds
        ckp = self.CheckForCheckpoints()
        acc = {}
        LastIdentity = ""
        if(ckp):
            acc = ckp['acc']
            LastIdentity = ckp['id']
        else:
            #create checkpoint file
            for t in Thresholds:
                acc[t] = {}
                for sim in self.Sensibility:
                        acc[t][sim] = {}
                        for elem in self.MinIDs:
                            acc[t][sim][elem] = {}
                            for w in self.FaceWeights:
                                acc[t][sim][elem][w] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            LastIdentity = ""

        for identity in tqdm(os.listdir(self.TrainDs), "evaluating images",position=0, leave=True):
            if(identity <= LastIdentity):
                continue
            
            realRef = os.path.join(self.TrainDs, identity, 'real')
            fakeFolder = os.path.join(self.TrainDs, identity, 'fake')
            acc =  self.ProcessIdentity(realRef, fakeFolder, acc, Thresholds)
            #save the checkpoint

            self.Save(acc, identity)      

        #remove checkpoints
        #os.remove(os.path.join(self.FOLDER, self.FILE)) 
        self.PlotResults(acc, Thresholds)


    def PlotResults(self, acc, THRESHOLD):
        best = {'threshold': 0, 'sim': 0, 'elem': 0, 'f1': 0, 'accuracy': 0, 'weight': 0}
        bestData = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for threshold in THRESHOLD:
            for sim in self.Sensibility:
                for elem in self.MinIDs:
                    for w in self.FaceWeights:
                        TP = acc[threshold][sim][elem][w]['TP']
                        TN = acc[threshold][sim][elem][w]['TN']
                        FP = acc[threshold][sim][elem][w]['FP']
                        FN = acc[threshold][sim][elem][w]['FN']
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
                            best['weight'] = w

        # compute the best data for a given threshold so to plot the ROC curve
        tpr_list = []
        fpr_list = []
        for threshold in THRESHOLD:
            _best = {'f1':0, 'TP':0, 'TN':0, 'FP':0, 'FN':0}
            for sim in self.Sensibility:
                for elem in self.MinIDs:
                    for w in self.FaceWeights:
                        TP = acc[threshold][sim][elem][w]['TP']
                        TN = acc[threshold][sim][elem][w]['TN']
                        FP = acc[threshold][sim][elem][w]['FP']
                        FN = acc[threshold][sim][elem][w]['FN']
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
        bw = best['weight']
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
        print("The best weight is: ", bw)
        input("Press Enter to continue...")
        plt.show()

    def Validate(self, threshold, sensitivity, weight, ids):
        acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for identity in tqdm(os.listdir(self.ValDs), "evaluating images",position=1, leave=True):
            rep = {}
            repFake = {}
            clipRep = {}
            clipRepFake = {}
            realRef = os.path.join(self.ValDs, identity, 'real')
            fakeFolder = os.path.join(self.ValDs, identity, 'fake')

            for real in os.listdir(realRef):
                rep[real] = self.Represent(os.path.join(realRef, real))
                clipRep[real] = self.ClipRepresent(os.path.join(realRef, real))
            for fake in os.listdir(fakeFolder):
                repFake[fake] = self.Represent(os.path.join(fakeFolder, fake))
                clipRepFake[fake] = self.ClipRepresent(os.path.join(fakeFolder, fake))

            #fake evaluation
            for fake in repFake.keys():
                distance = 0
                simList = []
                for real in rep.keys(): 
                    sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[real], clipRepFake[fake], dim=1).item() * (1 - float(weight))
                    distance = max(sim, distance)
                    simList.append(sim)
                
                if(distance > float(threshold) * float(sensitivity)):
                    acc['FP'] += 1
                else:
                    #in this case we require a min number of identities to be greater than the threshold
                    acceptables = 0
                    simList.sort(reverse=True)
                    for s in simList:
                        if(s > float(threshold)):
                            acceptables += 1
                            if(acceptables >= int(ids)):
                                acc['FP'] += 1
                                break
                    if(acceptables < int(ids)):
                        acc['TN'] += 1 

            #pick two random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
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
                    sim = torch.nn.functional.cosine_similarity(rep[real], rep[randomReal], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[real], clipRep[randomReal], dim=1).item() * (1 - float(weight))
                    distance = max(sim, distance)
                    simList.append(sim)
                
                if(distance > float(threshold) * float(sensitivity)):
                    acc['TP'] += 1
                else:
                    simList.sort(reverse=True)
                    for s in simList:
                        if(s > float(threshold)):
                            acceptables += 1
                            if(acceptables >= int(ids)):
                                acc['TP'] += 1
                                break
                    if(acceptables < int(ids)):
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
        
        print("Model: ", self.model)
        print("F1 score: ", f1)
        print("Accuracy: ", accuracy)
        input("Press Enter to continue...")
        plt.show()
        return

    def ProcessIdentity(self, realRef, fakeFolder, acc, THRESHOLD):
        rep = {}
        repFake = {}
        clipRep = {}
        clipRepFake = {}
        for real in os.listdir(realRef):
            rep[real] = self.Represent(os.path.join(realRef, real))      
            clipRep[real] = self.ClipRepresent(os.path.join(realRef, real))
        for fake in os.listdir(fakeFolder):
            repFake[fake] = self.Represent(os.path.join(fakeFolder, fake))
            clipRepFake[fake] = self.ClipRepresent(os.path.join(fakeFolder, fake))
            

        #fake evaluation
        for weight in self.FaceWeights:
            for fake in repFake.keys():
                distance = 0
                simList = []
                for real in rep.keys(): 
                    sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[real], clipRepFake[fake], dim=1).item() * (1 - float(weight))
                    simList.append(sim)
                    distance = max(sim, distance)
                simList.sort(reverse=True)

                
                for threshold in THRESHOLD:
                    for sim in self.Sensibility:
                        for elem in self.MinIDs:
                            if(distance > float(threshold) * float(sim)):
                                acc[threshold][sim][elem][weight]['FP'] += 1
                            else:
                                #in this case we require a min number of identities to be greater than the threshold
                                acceptables = 0
                                for s in simList:
                                    if(s > float(threshold)):
                                        acceptables += 1
                                        if(acceptables >= int(elem)):
                                            acc[threshold][sim][elem][weight]['FP'] += 1
                                            break
                                if(acceptables < int(elem)):
                                    acc[threshold][sim][elem][weight]['TN'] += 1

            #pick random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
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
                    sim = torch.nn.functional.cosine_similarity(rep[randomReal], rep[real], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[randomReal], clipRep[real], dim=1).item() * (1 - float(weight))
                    distance = max(sim, distance)
                    simList.append(sim)
                simList.sort(reverse=True)

                for threshold in THRESHOLD:
                    for sim in self.Sensibility:
                        for elem in self.MinIDs:
                            if(distance > float(threshold) * float(sim)):
                                acc[threshold][sim][elem][weight]['TP'] += 1
                            else:
                                #in this case we require a min number of identities to be greater than the threshold
                                acceptables = 0
                                for s in simList:
                                    if(s > float(threshold)):
                                        acceptables += 1
                                        if(acceptables >= int(elem)):
                                            acc[threshold][sim][elem][weight]['TP'] += 1    
                                            break
                                if(acceptables < int(elem)):
                                    acc[threshold][sim][elem][weight]['FN'] += 1

        return acc




class TFDoubleModel(TFRecognition):
    FaceWeights = []
    Model2 = ""

    def __init__(self, model, model2, TrainDs, ValDs,FaceWeight):
        super().__init__(model, TrainDs, ValDs)
        self.FaceWeights = FaceWeight
        self.Model2 = model2
    

    def Represent(self, path, model):
        embedding_objs = DeepFace.represent(
        img_path = path,
        model_name= model,
        enforce_detection=False,
        )
        SUT_embeddings = embedding_objs[0]['embedding']
        return torch.FloatTensor(SUT_embeddings)
    

    def ProcessIdentity(self, realRef, fakeFolder, acc, THRESHOLD):
        rep = {}
        repFake = {}
        rep2 = {}
        repFake2 = {}

        for real in os.listdir(realRef):
            rep[real] = self.Represent(os.path.join(realRef, real), self.model)
            rep2[real] = self.Represent(os.path.join(realRef, real), self.Model2)     
        for fake in os.listdir(fakeFolder):
            repFake[fake] = self.Represent(os.path.join(fakeFolder, fake), self.model)
            repFake2[fake] = self.Represent(os.path.join(fakeFolder, fake), self.Model2)
            
        #fake evaluation
        for weight in self.FaceWeights:
            for fake in repFake.keys():
                distance = 0
                for real in rep.keys(): 
                    sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(rep2[real], repFake2[fake], dim=0)* (1 - float(weight))
                    distance = max(sim, distance)

                for threshold in THRESHOLD:
                    if(distance > float(threshold)):
                        acc[threshold][weight]['FP'] += 1
                    else:
                        acc[threshold][weight]['TN'] += 1 

            #pick random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
            i = 0
            for randomReal in rep.keys():
                if(i == toEval):
                    break
                i += 1
                distance = 0
                for real in rep.keys():
                    if(real == randomReal):
                        continue
                    sim = torch.nn.functional.cosine_similarity(rep[randomReal], rep[real], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(rep2[randomReal], rep2[real], dim=0) * (1 - float(weight))
                    distance = max(sim, distance)
                
                for threshold in THRESHOLD:
                    if(distance > float(threshold)):
                        acc[threshold][weight]['TP'] += 1
                    else:
                        acc[threshold][weight]['FN'] += 1  

        return acc
                    
    def train(self, Thresholds):
        if(self.TrainDs == "" or self.ValDs == ""):
            print("No dataset provided")
            return
        elif(self.model == ""):
            print("No model provided")
            return
        if(not isinstance(Thresholds, list)):
            print("Thresholds should be a list")
            return
        
        #estimate the thresholds
        ckp = self.CheckForCheckpoints()
        acc = {}
        LastIdentity = ""
        if(ckp):
            acc = ckp['acc']
            LastIdentity = ckp['id']
        else:
            #create checkpoint file
            for t in Thresholds:
                acc[t] = {}
                for w in self.FaceWeights:
                    acc[t][w] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            LastIdentity = ""

        for identity in tqdm(os.listdir(self.TrainDs), "evaluating images",position=0, leave=True):
            if(identity <= LastIdentity):
                continue
            
            realRef = os.path.join(self.TrainDs, identity, 'real')
            fakeFolder = os.path.join(self.TrainDs, identity, 'fake')
            acc =  self.ProcessIdentity(realRef, fakeFolder, acc, Thresholds)
            #save the checkpoint

            self.Save(acc, identity)      

        #remove checkpoints
        #os.remove(os.path.join(self.FOLDER, self.FILE)) 
        self.PlotResults(acc, Thresholds)

    def PlotResults(self, acc, THRESHOLD):
        best = {'threshold': 0, 'weight': 0, 'f1': 0, 'accuracy': 0}
        bestData = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
 
        for t in THRESHOLD:
            for w in self.FaceWeights:
                FP = acc[t][w]['FP']
                FN = acc[t][w]['FN']
                TP = acc[t][w]['TP']     
                TN = acc[t][w]['TN']

                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                
                print(FP, FN, TP, TN)

                if(f1 > best['f1']):
                    bestData['TP'] = TP
                    bestData['TN'] = TN
                    bestData['FP'] = FP
                    bestData['FN'] = FN
                    best['threshold'] = t
                    best['weight'] = w
                    best['f1'] = f1
                    best['accuracy'] = (TP + TN) / (TP + TN + FP + FN)


        # Plot the diagonal line (random classifier)
        plt.figure(figsize=(12, 5))

        # Add labels and title
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        
        #plot the confusion matrix
        tf1 = best['f1']
        acc = best['accuracy']
        w = best['weight']
        t = best['threshold']

        FP = bestData['FP']
        FN = bestData['FN']
        TP = bestData['TP']     
        TN = bestData['TN']

        cm = np.array([[TN, FP], [FN, TP]])
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
        print("Model: ", self.model)
        print("Threshold:", round(float(t), 4))
        print("F1 score: ", round(tf1, 4))
        print("Accuracy: ", round(acc, 4))
        print("Weight: ", round(float(w), 4))
        #print("The treshold maximizing Accuracy is ", tacc, " with F1 score of ", f1[tacc], " and Accuracy of ", acc[tacc])
        input("Press Enter to continue...")
        plt.show()

    def Validate(self, threshold, weight):
        acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for identity in tqdm(os.listdir(self.ValDs), "evaluating images",position=1, leave=True):
            rep = {}
            repFake = {}
            rep2 = {}
            repFake2 = {}

            realRef = os.path.join(self.ValDs, identity, 'real')
            fakeFolder = os.path.join(self.ValDs, identity, 'fake')

            for real in os.listdir(realRef):
                rep[real] = self.Represent(os.path.join(realRef, real), self.model)
                rep2[real] = self.Represent(os.path.join(realRef, real), self.Model2)     
            for fake in os.listdir(fakeFolder):
                repFake[fake] = self.Represent(os.path.join(fakeFolder, fake), self.model)
                repFake2[fake] = self.Represent(os.path.join(fakeFolder, fake), self.Model2)
            

            #fake evaluation
            for fake in repFake.keys():
                distance = 0
                simList = []
                for real in rep.keys(): 
                    sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(rep2[real], repFake2[fake], dim=0) * (1 - float(weight))
                    distance = max(sim, distance)
                    simList.append(sim)
                
                if(distance > float(threshold)):
                    acc['FP'] += 1
                else:
                    acc['TN'] += 1


            #pick two random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
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
                    sim = torch.nn.functional.cosine_similarity(rep[real], rep[randomReal], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(rep2[real], rep2[randomReal], dim=0) * (1 - float(weight))
                    distance = max(sim, distance)
                    simList.append(sim)
                
                if(distance > float(threshold)):
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
        plt.title(f'Confusion Matrix (Threshold = {threshold})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        print("Model: ", self.model)
        print("F1 score: ", f1)
        print("Accuracy: ", accuracy)
        input("Press Enter to continue...")
        plt.show()
        return
    


class TFSensDoubleRecognition(TFRecognition):
    FaceWeights = []
    Model2 = ""
    def __init__(self, model, TrainDs, ValDs,FaceWeight, Sensibility, MinIDs, Model2):
        super().__init__(model, TrainDs, ValDs)
        self.FaceWeights = FaceWeight
        self.Sensibility = Sensibility
        self.MinIDs = MinIDs
        self.Model2 = Model2


    def Represent(self, path, model):
        embedding_objs = DeepFace.represent(
        img_path = path,
        model_name= model,
        enforce_detection=False,
        )
        SUT_embeddings = embedding_objs[0]['embedding']
        return torch.FloatTensor(SUT_embeddings)

    
    def train(self, Thresholds):
        if(self.TrainDs == "" or self.ValDs == ""):
            print("No dataset provided")
            return
        elif(self.model == ""):
            print("No model provided")
            return
        if(not isinstance(Thresholds, list)):
            print("Thresholds should be a list")
            return
        
        #estimate the thresholds
        ckp = self.CheckForCheckpoints()
        acc = {}
        LastIdentity = ""
        if(ckp):
            acc = ckp['acc']
            LastIdentity = ckp['id']
        else:
            #create checkpoint file
            for t in Thresholds:
                acc[t] = {}
                for sim in self.Sensibility:
                        acc[t][sim] = {}
                        for elem in self.MinIDs:
                            acc[t][sim][elem] = {}
                            for w in self.FaceWeights:
                                acc[t][sim][elem][w] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            LastIdentity = ""

        for identity in tqdm(os.listdir(self.TrainDs), "evaluating images",position=0, leave=True):
            if(identity <= LastIdentity):
                continue
            
            realRef = os.path.join(self.TrainDs, identity, 'real')
            fakeFolder = os.path.join(self.TrainDs, identity, 'fake')
            acc =  self.ProcessIdentity(realRef, fakeFolder, acc, Thresholds)
            #save the checkpoint

            self.Save(acc, identity)      

        #remove checkpoints
        #os.remove(os.path.join(self.FOLDER, self.FILE)) 
        self.PlotResults(acc, Thresholds)


    def PlotResults(self, acc, THRESHOLD):
        best = {'threshold': 0, 'sim': 0, 'elem': 0, 'f1': 0, 'accuracy': 0, 'weight': 0}
        bestData = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for threshold in THRESHOLD:
            for sim in self.Sensibility:
                for elem in self.MinIDs:
                    for w in self.FaceWeights:
                        TP = acc[threshold][sim][elem][w]['TP']
                        TN = acc[threshold][sim][elem][w]['TN']
                        FP = acc[threshold][sim][elem][w]['FP']
                        FN = acc[threshold][sim][elem][w]['FN']
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
                            best['weight'] = w

        # compute the best data for a given threshold so to plot the ROC curve
        tpr_list = []
        fpr_list = []
        for threshold in THRESHOLD:
            _best = {'f1':0, 'TP':0, 'TN':0, 'FP':0, 'FN':0}
            for sim in self.Sensibility:
                for elem in self.MinIDs:
                    for w in self.FaceWeights:
                        TP = acc[threshold][sim][elem][w]['TP']
                        TN = acc[threshold][sim][elem][w]['TN']
                        FP = acc[threshold][sim][elem][w]['FP']
                        FN = acc[threshold][sim][elem][w]['FN']
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
        bw = best['weight']
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
        print("The best weight is: ", bw)
        input("Press Enter to continue...")
        plt.show()

    def Validate(self, threshold, sensitivity, weight, ids):
        acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for identity in tqdm(os.listdir(self.ValDs), "evaluating images",position=1, leave=True):
            rep = {}
            repFake = {}
            clipRep = {}
            clipRepFake = {}
            realRef = os.path.join(self.ValDs, identity, 'real')
            fakeFolder = os.path.join(self.ValDs, identity, 'fake')

            for real in os.listdir(realRef):
                rep[real] = self.Represent(os.path.join(realRef, real),self.model)
                clipRep[real] = self.Represent(os.path.join(realRef, real), self.Model2)
            for fake in os.listdir(fakeFolder):
                repFake[fake] = self.Represent(os.path.join(fakeFolder, fake), self.model)
                clipRepFake[fake] = self.Represent(os.path.join(fakeFolder, fake),self.Model2)

            #fake evaluation
            for fake in repFake.keys():
                distance = 0
                simList = []
                for real in rep.keys(): 
                    sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[real], clipRepFake[fake], dim=0) * (1 - float(weight))
                    distance = max(sim, distance)
                    simList.append(sim)
                
                if(distance > float(threshold) * float(sensitivity)):
                    acc['FP'] += 1
                else:
                    #in this case we require a min number of identities to be greater than the threshold
                    acceptables = 0
                    simList.sort(reverse=True)
                    for s in simList:
                        if(s > float(threshold)):
                            acceptables += 1
                            if(acceptables >= int(ids)):
                                acc['FP'] += 1
                                break
                    if(acceptables < int(ids)):
                        acc['TN'] += 1 

            #pick two random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
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
                    sim = torch.nn.functional.cosine_similarity(rep[real], rep[randomReal], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[real], clipRep[randomReal], dim=0) * (1 - float(weight))
                    distance = max(sim, distance)
                    simList.append(sim)
                
                if(distance > float(threshold) * float(sensitivity)):
                    acc['TP'] += 1
                else:
                    simList.sort(reverse=True)
                    for s in simList:
                        if(s > float(threshold)):
                            acceptables += 1
                            if(acceptables >= int(ids)):
                                acc['TP'] += 1
                                break
                    if(acceptables < int(ids)):
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
        
        print("Model: ", self.model)
        print("F1 score: ", f1)
        print("Accuracy: ", accuracy)
        input("Press Enter to continue...")
        plt.show()
        return

    def ProcessIdentity(self, realRef, fakeFolder, acc, THRESHOLD):
        rep = {}
        repFake = {}
        clipRep = {}
        clipRepFake = {}
        for real in os.listdir(realRef):
            rep[real] = self.Represent(os.path.join(realRef, real), self.model)      
            clipRep[real] = self.Represent(os.path.join(realRef, real),self.Model2)
        for fake in os.listdir(fakeFolder):
            repFake[fake] = self.Represent(os.path.join(fakeFolder, fake), self.model)
            clipRepFake[fake] = self.Represent(os.path.join(fakeFolder, fake), self.Model2)
            

        #fake evaluation
        for weight in self.FaceWeights:
            for fake in repFake.keys():
                distance = 0
                simList = []
                for real in rep.keys(): 
                    sim = torch.nn.functional.cosine_similarity(rep[real], repFake[fake], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[real], clipRepFake[fake], dim=0) * (1 - float(weight))
                    simList.append(sim)
                    distance = max(sim, distance)
                simList.sort(reverse=True)

                
                for threshold in THRESHOLD:
                    for sim in self.Sensibility:
                        for elem in self.MinIDs:
                            if(distance > float(threshold) * float(sim)):
                                acc[threshold][sim][elem][weight]['FP'] += 1
                            else:
                                #in this case we require a min number of identities to be greater than the threshold
                                acceptables = 0
                                for s in simList:
                                    if(s > float(threshold)):
                                        acceptables += 1
                                        if(acceptables >= int(elem)):
                                            acc[threshold][sim][elem][weight]['FP'] += 1
                                            break
                                if(acceptables < int(elem)):
                                    acc[threshold][sim][elem][weight]['TN'] += 1

            #pick random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
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
                    sim = torch.nn.functional.cosine_similarity(rep[randomReal], rep[real], dim=0) * float(weight) + torch.nn.functional.cosine_similarity(clipRep[randomReal], clipRep[real], dim =0) * (1 - float(weight))
                    distance = max(sim, distance)
                    simList.append(sim)
                simList.sort(reverse=True)

                for threshold in THRESHOLD:
                    for sim in self.Sensibility:
                        for elem in self.MinIDs:
                            if(distance > float(threshold) * float(sim)):
                                acc[threshold][sim][elem][weight]['TP'] += 1
                            else:
                                #in this case we require a min number of identities to be greater than the threshold
                                acceptables = 0
                                for s in simList:
                                    if(s > float(threshold)):
                                        acceptables += 1
                                        if(acceptables >= int(elem)):
                                            acc[threshold][sim][elem][weight]['TP'] += 1    
                                            break
                                if(acceptables < int(elem)):
                                    acc[threshold][sim][elem][weight]['FN'] += 1

        return acc


class TFNModels(TFRecognition):
    FaceWeights = []
    model = []
    N = 0
    def __init__(self, model, TrainDs, ValDs, singleModelWeights):
        self.TrainDs  = TrainDs
        self.ValDs = ValDs
        self.model = model

        self.N = len(model)
        weights = [float(i) for i in singleModelWeights]
        all_combinations = itertools.product(weights, repeat=self.N)

        # Filter combinations where the sum of elements is 1
        valid_combinations = [combination for combination in all_combinations if sum(combination) == 1.0]

        # Print the valid combinations
        self.FaceWeights = valid_combinations
                
    

    def Represent(self, path, model):
        embedding_objs = DeepFace.represent(
        img_path = path,
        model_name= model,
        enforce_detection=False,
        )
        SUT_embeddings = embedding_objs[0]['embedding']
        return torch.FloatTensor(SUT_embeddings)
    

    def ProcessIdentity(self, realRef, fakeFolder, acc, THRESHOLD):
        rep = {}
        repFake = {}
        
        for real in os.listdir(realRef):
            rep[real] = {}
            for model in self.model:
                rep[real][model] = self.Represent(os.path.join(realRef, real), model) 
        for fake in os.listdir(fakeFolder):
            repFake[fake] = {}
            for model in self.model:
                repFake[fake][model] = self.Represent(os.path.join(fakeFolder, fake), model)

            
        #pick random real images and perform evaluation
        toEval = self.count_files_in_directory(fakeFolder)
        
        for fake in repFake.keys():           
            for weight in self.FaceWeights:
                index = self.FaceWeights.index(weight)
                distance = 0
                for real in rep.keys(): 
                    sim = 0
                    for i in range(self.N):
                        sim += torch.nn.functional.cosine_similarity(rep[real][self.model[i]], repFake[fake][self.model[i]], dim=0) * float(weight[i])
                    distance = max(sim, distance)

                
                for threshold in THRESHOLD:
                    if(distance > float(threshold)):
                        acc[threshold][str(index)]['FP'] += 1
                    else:
                        acc[threshold][str(index)]['TN'] += 1 

        
        i = 0 
        for randomReal in rep.keys():
            if(i == toEval):
                break
            i += 1            
            distance = 0
            for weight in self.FaceWeights:
                index = self.FaceWeights.index(weight)          
                for real in rep.keys():
                    if(real == randomReal):
                        continue
                    sim = 0
                    for j in range(self.N):
                        sim += torch.nn.functional.cosine_similarity(rep[randomReal][self.model[j]], rep[real][self.model[j]], dim=0) * float(weight[j])    
                    distance = max(sim, distance)
                
                for threshold in THRESHOLD:
                    if(distance > float(threshold)):
                        acc[threshold][str(index)]['TP'] += 1
                    else:
                        acc[threshold][str(index)]['FN'] += 1  
        return acc
    
                    
    def train(self, Thresholds):
        if(self.TrainDs == "" or self.ValDs == ""):
            print("No dataset provided")
            return
        elif(self.model == ""):
            print("No model provided")
            return
        if(not isinstance(Thresholds, list)):
            print("Thresholds should be a list")
            return
        
        #estimate the thresholds
        ckp = self.CheckForCheckpoints()
        acc = {}
        LastIdentity = ""
        if(ckp):
            acc = ckp['acc']
            LastIdentity = ckp['id']
        else:
            #create checkpoint file
            for t in Thresholds:
                acc[t] = {}
                for w in self.FaceWeights:
                    index = self.FaceWeights.index(w)
                    acc[t][str(index)] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            LastIdentity = ""

        for identity in tqdm(os.listdir(self.TrainDs), "evaluating images",position=0, leave=True):
            if(identity <= LastIdentity):
                continue
            
            realRef = os.path.join(self.TrainDs, identity, 'real')
            fakeFolder = os.path.join(self.TrainDs, identity, 'fake')
            acc =  self.ProcessIdentity(realRef, fakeFolder, acc, Thresholds)

            #save the checkpoint

            self.Save(acc, identity)      

        #remove checkpoints
        #os.remove(os.path.join(self.FOLDER, self.FILE)) 
        self.PlotResults(acc, Thresholds)

    def PlotResults(self, acc, THRESHOLD):
        best = {'threshold': 0, 'weight': 0, 'f1': 0, 'accuracy': 0}
        bestData = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
 
        for t in THRESHOLD:
            for w in self.FaceWeights:
                index = self.FaceWeights.index(w)
                FP = acc[t][str(index)]['FP']
                FN = acc[t][str(index)]['FN']
                TP = acc[t][str(index)]['TP']     
                TN = acc[t][str(index)]['TN']

                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                

                if(f1 > best['f1']):
                    bestData['TP'] = TP
                    bestData['TN'] = TN
                    bestData['FP'] = FP
                    bestData['FN'] = FN
                    best['threshold'] = t
                    best['weight'] = w
                    best['f1'] = f1
                    best['accuracy'] = (TP + TN) / (TP + TN + FP + FN)


        # Plot the diagonal line (random classifier)
        plt.figure(figsize=(12, 5))

        # Add labels and title
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        
        #plot the confusion matrix
        tf1 = best['f1']
        acc = best['accuracy']
        w = best['weight']
        t = best['threshold']

        FP = bestData['FP']
        FN = bestData['FN']
        TP = bestData['TP']     
        TN = bestData['TN']

        cm = np.array([[TN, FP], [FN, TP]])
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
        print("Model: ", self.model)
        print("Threshold:", round(float(t), 4))
        print("F1 score: ", round(tf1, 4))
        print("Accuracy: ", round(acc, 4))
        print("Weight: ", w)
        input("Press Enter to continue...")
        plt.show()

    def Validate(self, threshold, weight):
        acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for identity in tqdm(os.listdir(self.ValDs), "evaluating images",position=1, leave=True):
            realRef = os.path.join(self.ValDs, identity, 'real')
            fakeFolder = os.path.join(self.ValDs, identity, 'fake')

            rep = {}
            repFake = {}
      
            for real in os.listdir(realRef):
                rep[real] = {}
                for model in self.model:
                    rep[real][model] = self.Represent(os.path.join(realRef, real), model) 
            for fake in os.listdir(fakeFolder):
                repFake[fake] = {}
                for model in self.model:
                    repFake[fake][model] = self.Represent(os.path.join(fakeFolder, fake), model)
            

            #fake evaluation
            for fake in repFake.keys():
                distance = 0
                for real in rep.keys(): 
                    sim = 0
                    for i in range(self.N):
                        sim += torch.nn.functional.cosine_similarity(rep[real][self.model[i]], repFake[fake][self.model[i]], dim=0) * float(weight[i])
                    distance = max(sim, distance)
                
                if(distance > float(threshold)):
                    acc['FP'] += 1
                else:
                    acc['TN'] += 1

            #pick two random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
            i = 0
            for randomReal in rep.keys():
                if(i == toEval):
                    break
                i += 1
                distance = 0
                for real in rep.keys():
                    if(real == randomReal):
                        continue
                    sim = 0
                    for j in range(self.N):
                        sim += torch.nn.functional.cosine_similarity(rep[randomReal][self.model[j]], rep[real][self.model[j]], dim=0) * float(weight[j])
                    distance = max(sim, distance)
                
                if(distance > float(threshold)):
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
        plt.title(f'Confusion Matrix (Threshold = {threshold})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        print("Model: ", self.model)
        print("F1 score: ", f1)
        print("Accuracy: ", accuracy)
        input("Press Enter to continue...")
        plt.show()
        return
    


class TFNModelsSens(TFRecognition):
    FaceWeights = []
    model = []
    N = 0
    sensibility = []
    MinIDs = []
    def __init__(self, model, TrainDs, ValDs, singleModelWeights, Sensibility, MinIDs):
        self.TrainDs  = TrainDs
        self.ValDs = ValDs
        self.model = model
        self.sensibility = Sensibility
        self.MinIDs = MinIDs

        self.N = len(model)
        weights = [float(i) for i in singleModelWeights]
        all_combinations = itertools.product(weights, repeat=self.N)

        # Filter combinations where the sum of elements is 1
        valid_combinations = [combination for combination in all_combinations if sum(combination) == 1.0]

        # Print the valid combinations
        self.FaceWeights = valid_combinations
                
    

    def Represent(self, path, model):
        embedding_objs = DeepFace.represent(
        img_path = path,
        model_name= model,
        enforce_detection=False,
        )
        SUT_embeddings = embedding_objs[0]['embedding']
        return torch.FloatTensor(SUT_embeddings)
    

    def ProcessIdentity(self, realRef, fakeFolder, acc, THRESHOLD):
        rep = {}
        repFake = {}
        
        for real in os.listdir(realRef):
            rep[real] = {}
            for model in self.model:
                rep[real][model] = self.Represent(os.path.join(realRef, real), model) 
        for fake in os.listdir(fakeFolder):
            repFake[fake] = {}
            for model in self.model:
                repFake[fake][model] = self.Represent(os.path.join(fakeFolder, fake), model)

        #pick random real images and perform evaluation
        toEval = self.count_files_in_directory(fakeFolder)
        
        for fake in tqdm(repFake.keys(), desc="FAKES", position=1, leave=True):         
            simList = []

            for weight in self.FaceWeights:
                index = self.FaceWeights.index(weight)
                simList = []
                maxSim = 0
                for real in rep.keys(): 
                    sim = 0
                    for j in range(self.N):
                        sim += torch.nn.functional.cosine_similarity(rep[real][self.model[j]], repFake[fake][self.model[j]], dim=0) * float(weight[j])
                    simList.append(sim)             
                    maxSim = max(sim, maxSim)
                
                for threshold in THRESHOLD:
                    for sim in self.sensibility:
                        for ids in self.MinIDs:
                            if(maxSim > float(threshold) * float(sim)):
                                acc[threshold][sim][ids][str(index)]['FP'] += 1
                            else:
                                #in this case we require a min number of identities to be greater than the threshold
                                acceptables = 0
                                simList.sort(reverse=True)
                                for s in simList:
                                    if(s > float(threshold)):
                                        acceptables += 1
                                        if(acceptables >= int(ids)):
                                            acc[threshold][sim][ids][str(index)]['FP'] += 1
                                            break
                                if(acceptables < int(ids)):
                                    acc[threshold][sim][ids][str(index)]['TN'] += 1
        
        i = 0 
        for randomReal in tqdm(rep.keys(), desc="REALS", position=1, leave=True):
            if(i == toEval):
                break
            i += 1            
            simList = []
            for weight in self.FaceWeights:
                index = self.FaceWeights.index(weight) 
                simList = []
                maxSim = 0
                for real in rep.keys():
                    if(real == randomReal):
                        continue
                    sim = 0
                    for j in range(self.N):
                        sim += torch.nn.functional.cosine_similarity(rep[randomReal][self.model[j]], rep[real][self.model[j]], dim=0) * float(weight[j])    
                    simList.append(sim)             
                    maxSim = max(sim, maxSim)
                
                for threshold in THRESHOLD:
                    for sim in self.sensibility:
                        for ids in self.MinIDs:
                            if(maxSim > float(threshold) * float(sim)):
                                acc[threshold][sim][ids][str(index)]['TP'] += 1
                            else:
                                #in this case we require a min number of identities to be greater than the threshold
                                acceptables = 0
                                simList.sort(reverse=True)
                                for s in simList:
                                    if(s > float(threshold)):
                                        acceptables += 1
                                        if(acceptables >= int(ids)):
                                            acc[threshold][sim][ids][str(index)]['TP'] += 1
                                            break
                                if(acceptables < int(ids)):
                                    acc[threshold][sim][ids][str(index)]['FN'] += 1
        

        return acc
    
                    
    def train(self, Thresholds):
        if(self.TrainDs == "" or self.ValDs == ""):
            print("No dataset provided")
            return
        elif(self.model == ""):
            print("No model provided")
            return
        if(not isinstance(Thresholds, list)):
            print("Thresholds should be a list")
            return
        
        #estimate the thresholds
        ckp = self.CheckForCheckpoints()
        acc = {}
        LastIdentity = ""
        if(ckp):
            acc = ckp['acc']
            LastIdentity = ckp['id']
            print("CHECKPOINT FOUND, RESUMING FROM IDENTITY: ", LastIdentity)
        else:
            #create checkpoint file
            for t in Thresholds:
                acc[t] = {}
                for s in self.sensibility:
                    acc[t][s] = {}
                    for id in self.MinIDs:
                        acc[t][s][id] = {}
                        for w in self.FaceWeights:
                            index = self.FaceWeights.index(w)
                            acc[t][s][id][str(index)] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            LastIdentity = ""

        for identity in tqdm(os.listdir(self.TrainDs), "evaluating images",position=0, leave=True):
            if(identity <= LastIdentity):
                continue
            
            realRef = os.path.join(self.TrainDs, identity, 'real')
            fakeFolder = os.path.join(self.TrainDs, identity, 'fake')
            acc =  self.ProcessIdentity(realRef, fakeFolder, acc, Thresholds)

            #save the checkpoint

            self.Save(acc, identity)      

        #remove checkpoints
        #os.remove(os.path.join(self.FOLDER, self.FILE)) 
        self.PlotResults(acc, Thresholds)

    def PlotResults(self, acc, THRESHOLD):
        best = {'threshold': 0, 'weight': 0, 'f1': 0, 'accuracy': 0,'minId': 0, 'Ss': 0}
        bestData = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
 
        for t in THRESHOLD:
            for s in self.sensibility:
                for w in self.FaceWeights:
                    for id in self.MinIDs:
                        for w in self.FaceWeights:
                            index = self.FaceWeights.index(w)
                            FP = acc[t][s][id][str(index)]['FP']
                            FN = acc[t][s][id][str(index)]['FN']
                            TP = acc[t][s][id][str(index)]['TP']     
                            TN = acc[t][s][id][str(index)]['TN']

                            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                            

                            if(f1 > best['f1']):
                                bestData['TP'] = TP
                                bestData['TN'] = TN
                                bestData['FP'] = FP
                                bestData['FN'] = FN
                                best['threshold'] = t
                                best['weight'] = w
                                best['f1'] = f1
                                best['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
                                best['minId'] = id
                                best['Ss'] = s


        # Plot the diagonal line (random classifier)
        plt.figure(figsize=(12, 5))

        # Add labels and title
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        
        #plot the confusion matrix
        tf1 = best['f1']
        acc = best['accuracy']
        w = best['weight']
        t = best['threshold']
    
        FP = bestData['FP']
        FN = bestData['FN']
        TP = bestData['TP']     
        TN = bestData['TN']

        cm = np.array([[TN, FP], [FN, TP]])
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
        print("Model: ", self.model)
        print("Threshold:", round(float(t), 4))
        print("F1 score: ", round(tf1, 4))
        print("Accuracy: ", round(acc, 4))
        print("Weight: ", w)
        print("Min Identities: ", best['minId'])
        print("Sensibility: ", best['Ss'])
        input("Press Enter to continue...")
        plt.show()

    def Validate(self, threshold, weights, ids, sens):
        acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for identity in tqdm(os.listdir(self.ValDs), "evaluating images",position=1, leave=True):
            realRef = os.path.join(self.ValDs, identity, 'real')
            fakeFolder = os.path.join(self.ValDs, identity, 'fake')

            rep = {}
            repFake = {}
            
            for real in os.listdir(realRef):
                rep[real] = {}
                for model in self.model:
                    rep[real][model] = self.Represent(os.path.join(realRef, real), model) 
            for fake in os.listdir(fakeFolder):
                repFake[fake] = {}
                for model in self.model:
                    repFake[fake][model] = self.Represent(os.path.join(fakeFolder, fake), model)

            #pick random real images and perform evaluation
            toEval = self.count_files_in_directory(fakeFolder)
            
            for fake in tqdm(repFake.keys(), desc="FAKES", position=1, leave=True):         
                simList = []
                maxSim = 0
                for real in rep.keys(): 
                    sim = 0
                    for j in range(self.N):
                        sim += torch.nn.functional.cosine_similarity(rep[real][self.model[j]], repFake[fake][self.model[j]], dim=0) * float(weights[j])
                    simList.append(sim)             
                    maxSim = max(sim, maxSim)
                    

                if(maxSim > float(threshold) * float(sens)):
                    acc['FP'] += 1
                else:
                    #in this case we require a min number of identities to be greater than the threshold
                    acceptables = 0
                    simList.sort(reverse=True)
                    for s in simList:
                        if(s > float(threshold)):
                            acceptables += 1
                            if(acceptables >= int(ids)):
                                acc['FP'] += 1
                                break
                    if(acceptables < int(ids)):
                        acc['TN'] += 1
            
            i = 0 
            for randomReal in tqdm(rep.keys(), desc="REALS", position=1, leave=True):
                if(i == toEval):
                    break
                i += 1            
                simList = []
                maxSim = 0
                for real in rep.keys():
                    if(real == randomReal):
                        continue
                    sim = 0
                    for j in range(self.N):
                        sim += torch.nn.functional.cosine_similarity(rep[randomReal][self.model[j]], rep[real][self.model[j]], dim=0) * float(weights[j])    
                    simList.append(sim)             
                    maxSim = max(sim, maxSim)
                    
                if(maxSim > float(threshold) * float(sens)):
                    acc['TP'] += 1
                else:
                    #in this case we require a min number of identities to be greater than the threshold
                    acceptables = 0
                    simList.sort(reverse=True)
                    for s in simList:
                        if(s > float(threshold)):
                            acceptables += 1
                            if(acceptables >= int(ids)):
                                acc['TP'] += 1
                                break
                    if(acceptables < int(ids)):
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
        
        print("Model: ", self.model)
        print("F1 score: ", f1)
        print("Accuracy: ", accuracy)
        input("Press Enter to continue...")
        plt.show()
        return



