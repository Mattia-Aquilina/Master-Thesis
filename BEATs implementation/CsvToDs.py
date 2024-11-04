import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
from io import BytesIO

directory = "./Dataset_tools/Csv/"
saveDir = "./Dataset_tools/Dataset/"

models = ['model_gen0', 'model_gen1', 'model_gen2', 'model_gen3']
images = ['image_gen0.bytes', 'image_gen1.bytes', 'image_gen2.bytes', 'image_gen3.bytes']

paths = ['filepath_gen0', 'filepath_gen1','filepath_gen2', 'filepath_gen3' ]

width = ['width_gen0', 'width_gen1', 'width_gen2', 'width_gen3']
height = ['height_gen0', 'height_gen1', 'height_gen2', 'height_gen3']

# Assicurati che la directory di salvataggio esista
os.makedirs(saveDir, exist_ok=True)
counter = 0

for file in tqdm(os.listdir(directory), desc="Processing CSV files"):
    filename = os.fsdecode(directory + file)
    df = pd.read_csv(directory + file, delimiter=',')   
    for index, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):    
        counter +=1
        for i in range(len(models)):
            _width = getattr(row, width[i])
            _height = getattr(row, height[i])

            img_str = getattr(row, images[i])
            img_str = img_str[2:-1]

            b= bytes(img_str, 'latin1').decode('unicode_escape').encode('latin1')


            dirName =saveDir + getattr(row, models[i]).replace('/', '_')


            if not os.path.exists(dirName): 
                os.makedirs(dirName) 


            imagePath = getattr(row, paths[i])

            if(imagePath.endswith('.jpg')):
                newFile = open(os.path.join(dirName, str(counter) + ".jpg"), "wb")
                newFile.write(b)
            else:
                newFile = open(os.path.join(dirName, str(counter) + ".png"), "wb")
                newFile.write(b)
                img = Image.open(os.path.join(dirName, str(counter) + ".png"))
                img.save(os.path.join(dirName, str(counter) + ".jpg"))
                img.close()

    
            
            






            









