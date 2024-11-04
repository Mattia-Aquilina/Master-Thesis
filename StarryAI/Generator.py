import requests
from PIL import Image
import base64
from io import BytesIO
import os, random
import tqdm
import time
from pathlib import Path

KEY = "xjO_36phyS6ZD2SOovLssn9ryGmsBg"
datasetFolder = "GoogleDataset/GDatasetSplit"	
newDatasetFolder = "GoogleDataset/GDatasetSplit"
GenerationCost = 0.0096
NAME_OFFSET = 2
TARGET_NUMBER = 15
#prompts = [" getting interviewed", " playing a musical instrument", " reading a book"]

def Authenticate():
    url = "https://api.starryai.com/user/"

    headers = {
        "X-API-Key": KEY,  # Replace with your actual API key
        "accept": "application/json"
    }

    # Send GET request to the API
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        balance  =  response.json()["balance"]
        response.close()
        return balance >= GenerationCost
    else:
        print(f"Failed to fetch user data, status code: {response.status_code}")
        response.close()
        return False

def GenerateImage(prompt, encodedImg):
    url = "https://api.starryai.com/creations/"

    payload = {
        "model": "realvisxl",
        "highResolution": False,
        "images": 1,
        "steps": 20,
        "initialImageMode": "color",
        "initialImageEncoded": encodedImg,
        "prompt": prompt
    }
    headers = {
        "X-API-Key": KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    request = requests.post(url, json=payload, headers=headers)
    if(request.status_code == 200):
        retrunValue = request.json()['id']
        request.close()
        return retrunValue
    else:
        text = request.text
        request.close()
        print("Error: ", text)
    return 

def GetImage(imageId):
    url = "https://api.starryai.com/creations/" + str(imageId)
    headers = {"accept": "application/json", "X-API-Key": KEY,}

    response = requests.get(url, headers=headers)

    if(response.json()['status'] == "completed"):
        url = response.json()['images'][0]['url']
        response.close()
        return (True, url)
    else:
        response.close()
        return (False, None)


def Base64EncodeImage(image):
    with Image.open(image) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def download_image(image_url, file_name):
    # Send GET request to the image URL
    img_response = requests.get(image_url)

    # Check if the request was successful
    if img_response.status_code == 200:
        # Open a file in binary mode to write the image content
        with open(file_name, 'wb') as img_file:
            img_file.write(img_response.content)
        print(f"Image successfully downloaded and saved as {file_name}.")
    else:
        print(f"Failed to retrieve image. Status code: {img_response.status_code}")
   
def ProcessImage(imagePath, foldername, id):
    
    savePath = os.path.join(newDatasetFolder, os.path.join(foldername, "fake", "STR-"+ str(id) + ".jpg"))

    encoded = Base64EncodeImage(imagePath)
    prompt = foldername.replace("_", " ")
    
    generationId = GenerateImage(prompt + " portait.", encoded)

    #wait for generation to complete
    canDownload = False
    url = None
    while(canDownload == False):
        canDownload, url = GetImage(generationId)
        time.sleep(5)

    #the path will be newDataset/foldername/fake/imagename.jpg
    download_image(url, savePath)

    #url is now accessible
     

def AlphabeticalOrder(s1, s2):
    if(s1 < s2):
        return True
    else:
        return False



def ProcessDataset(lastFileOpened = ""): 
    for file in tqdm.tqdm(os.listdir(newDatasetFolder), desc="Processing Folders"):

        if(not Authenticate()):
            print("Not enough balance\n")
            print("!!!SAVE THIS VALUE!!! FILE TO PROCESS WAS: ", file, )
            return 
        
        if(lastFileOpened != "" and AlphabeticalOrder(file, lastFileOpened)):
            continue
        if(lastFileOpened != "" and file == lastFileOpened):
            continue

        pathReal = os.path.join(newDatasetFolder, os.path.join(file,"real"))
        #pick a random image from the real folder

        #count how many images are in the fake folder
        fakePath = os.path.join(newDatasetFolder, os.path.join(file,"fake"))
        num = sum(1 for real in Path(fakePath).iterdir() if real.is_file())
        if(num >= TARGET_NUMBER):
            continue

        for i in range(TARGET_NUMBER - num):
            id = NAME_OFFSET + num + i
            realImage = random.choice(os.listdir(pathReal))
            while(realImage.endswith(".png")):
                realImage = random.choice(os.listdir(pathReal))
            realImagePath = os.path.join(pathReal, realImage)
            ProcessImage(realImagePath, file, id)


def ProcessGoogleDataset(lastFileOpened = ""): 

    for file in tqdm.tqdm(os.listdir(newDatasetFolder), desc="Processing Folders"):
        if(lastFileOpened != "" and AlphabeticalOrder(file, lastFileOpened)):
            continue


        #count how many images are in the fake folder
        fakePath = os.path.join(newDatasetFolder, os.path.join(file,"fake"))
        pathReal = os.path.join(newDatasetFolder, os.path.join(file,"real"))
        num = sum(1 for real in Path(fakePath).iterdir() if real.is_file())
        if(num >= TARGET_NUMBER):
            continue

        for i in range(TARGET_NUMBER - num):
            if(not Authenticate()):
                print("Not enough balance\n")
                print("!!!SAVE THIS VALUE!!! FILE TO PROCESS WAS: ", file, )
                return 
        
            id = NAME_OFFSET + num + i
            realImage = random.choice(os.listdir(pathReal))
            realImagePath = os.path.join(pathReal, realImage)
            ProcessImage(realImagePath, file, id)


LFO = "Jennifer_Connelly"  

ProcessGoogleDataset()