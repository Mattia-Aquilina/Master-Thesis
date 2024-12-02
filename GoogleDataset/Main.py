import TrainingBase as tb

OLDER = "Checkpoints"
FILE = "ckpt.json"
#DATA
THRESHOLD = ['0.1', '0.12', '0.14', '0.16', '0.18', '0.2', '0.22', '0.24', '0.26', '0.28', '0.3', '0.32', '0.34', '0.36', '0.38', '0.4', '0.45', '0.5', '0.52', '0.54', '0.56', '0.58', '0.6', '0.62', '0.64', '0.66', '0.68', '0.7', '0.72', '0.74', '0.76', '0.78', '0.8', '0.82', '0.84', '0.86', '0.88', '0.9', '0.92', '0.94', '0.96', '0.98', '1']
#THRESHOLD = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5 ,16, 16.5, 17, 17.5 , 18, 18.5, 19, 19.5, 20, 20.5, 21] #from 1 to 30
WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
MIN_ID = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
Ss = ['1.05', '1.07','1.08','1.1' , '1.25','1.15','1.2', '1.25', '1.3' , '1.35']
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

trainer = tb.TFDoubleModel("Facenet", "GhostFaceNet",DATASET, VALDS, WEIGHTS)
#trainer = tb.TFSensRecognition("ArcFace", DATASET, VALDS, Ss, MIN_ID)
#trainer.train(THRESHOLD)
trainer.Validate(threshold=0.58, weight=0.5)