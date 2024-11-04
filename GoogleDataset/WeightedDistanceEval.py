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
#from EvaluateDataset import Represent, MODELS, count_files_in_directory, DATASET, VALDS, PlotResults

EuclideanWeight = .33
ManathanWeight = .33
# MahalanobisWeight = .2
CosineWeight = 0.33


def EuclideanDistance(a, b):
    return torch.dist(a, b, p=2)

def ManathanDistance(a, b):
    return torch.dist(a, b, p=1)

# def MahalanobisDistance(embedding_test, embedding_real):
#     embeddings = torch.stack([embedding_test, embedding_real])
#     mean_vec = torch.mean(embeddings, dim=0)

#     # Compute the covariance matrix of the real embeddings
#     cov_matrix = torch.cov(embeddings.T)  # Compute covariance matrix
#     inv_cov_matrix = torch.linalg.inv(cov_matrix)  # Inverse of the covariance matrix

#     # Compute Mahalanobis distance
#     diff = embedding_test - embedding_real
#     return torch.sqrt(torch.mm(torch.mm(diff.unsqueeze(0), inv_cov_matrix), diff.unsqueeze(1)))

def WeightedDistance(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0) * CosineWeight + EuclideanDistance(a, b) * EuclideanWeight + ManathanDistance(a, b) * ManathanWeight

# def ValidateDataset(model, threshold):
#     acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

#     for identity in tqdm(os.listdir(VALDS), "Validation",position=1, leave=True):
#         rep = {}
#         repFake = {}
#         realRef = os.path.join(VALDS, identity, 'real')
#         fakeFolder = os.path.join(VALDS, identity, 'fake')

#         for real in os.listdir(realRef):
#             rep[real] = Represent(os.path.join(realRef, real), model)
#         for fake in os.listdir(fakeFolder):
#             repFake[fake] = Represent(os.path.join(fakeFolder, fake), model)

#         #fake evaluation (for each fake image)
#         for fake in repFake.keys():
#             distance = 0
#             for real in rep.keys(): 
#                 sim = WeightedDistance(rep[real], repFake[fake]) 
#                 return
#                 distance = max(sim, distance)

            
#             if(distance >= threshold):
#                 acc['FP'] += 1
#             else:
#                 acc['TN'] += 1       
        
#         # Real Evaluation (for each real image)
#         toEval = count_files_in_directory(fakeFolder)
#         i = 0
#         for randomReal in rep.keys():
#             if(i >= toEval):
#                 break
#             i += 1
#             distance = 0
#             for real in rep.keys():
#                 if(real == randomReal):
#                     continue
#                 sim = WeightedDistance(rep[real], rep[randomReal])
#                 distance = max(sim, distance)

                
#             if(distance >= threshold):
#                 acc['TP'] += 1
#             else:
#                 acc['FN'] += 1      
        
#     PlotResults(acc['FP'],acc['FN'] ,acc['TP'], acc['TN'],model, threshold)

