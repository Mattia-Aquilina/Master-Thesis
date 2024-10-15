import requests
from PIL import Image
import base64
from io import BytesIO
import os, random
import tqdm
import time
import getpass

KEY = "xjO_36phyS6ZD2SOovLssn9ryGmsBg"
dataset = "GoogleDataset/GDatasetSplit"	
LAST_IDENTITY = "Abigail_Spencer"
prompts = [" getting interviewed by a journalist", " playing a musical instrument", " reading a book"]

def QueryImage(prompt, path, realimage):
    response = requests.post(
    f"https://api.stability.ai/v2beta/stable-image/control/structure",
    headers={
        "authorization": f"Bearer sk-Et05BJxLofme4Ks16kyD9EYhTbUSDY8Z0KgnRoNKsO5CcGA3",
        "accept": "image/*"
    },
    files={
        "image": open(realimage, "rb")
    },
    data={
        "prompt": prompt,
        "control_strength": 0.7,
        "output_format": "webp"
    },
)

    if response.status_code == 200:
        with open(path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(str(response.json()))


def Generate():
    found = False
    for identity in tqdm.tqdm(os.listdir(dataset), desc="Processing folders"):
        
        if(LAST_IDENTITY != "" and not found and LAST_IDENTITY != identity):           
            continue
        elif(LAST_IDENTITY != "" and LAST_IDENTITY == identity and not found):
            found = True
            continue
        #pick a random real iamge
        realPath = os.path.join(dataset, identity, "real")
        realImage = random.choice(os.listdir(realPath))

        #pick a random prompt
        print("processing " + identity)
        prompt = identity.replace("_", " ") + random.choice(prompts)
        QueryImage(prompt, os.path.join(dataset, identity, "fake", "generated.jpg"), os.path.join(realPath, realImage))
        


Generate()