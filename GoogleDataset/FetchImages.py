import requests
import csv
import os 

CURRENT_FOLDER = "GoogleDataset/"
NAMES_CSV = "celebrity_data.csv"
# Dataset
NAME = "GDatasetSplit"
# Google API Key
#API_KEY = "AIzaSyAf_MGbyweteo6ydxDIbE59ACdUj5Fcfa4" #mattia.eagle
API_KEY = "AIzaSyDhjpuxe6aZ2PdzMs3c7mCJWEi6ygUizqY" #studenti.uniroma1
SEARCH_ENGINE_ID = "15f14800fb38f4f76"
URL = "https://www.googleapis.com/customsearch/v1"
REQUESTS = 57
START = 10
name_offset = 31
LAST_IDENTITY = "Mitch_Albom"



def download_image(image_url, file_name):
    # Send GET request to the image URL
    try:
        img_response = requests.get(image_url)
    except Exception as request_exception:
        print("Failed to retrieve image" + request_exception)
        return None
    # Check if the request was successful
    if img_response.status_code == 200:
        # Open a file in binary mode to write the image content
        with open(file_name, 'wb') as img_file:
            img_file.write(img_response.content)
        print(f"Image successfully downloaded and saved as {file_name}.")
        img_response.close()
    else:
        print(f"Failed to retrieve image. Status code: {img_response.status_code}")
        img_response.close()

def FormatName(name):
    for i in range(0, len(name)-1):
        if(i == 0):
            identity = name[i].replace(".","")
        else:
            identity += "_" + name[i].replace(".","")

    return identity

if(not os.path.exists(os.path.join(CURRENT_FOLDER, NAME))):
    os.mkdir((os.path.join(CURRENT_FOLDER, NAME)))

# def fromCSV():
#     reqs = 0
#     with open(os.path.join(CURRENT_FOLDER, NAMES_CSV)) as csvfile: 
#         spamreader = csv.reader(csvfile, delimiter=',', )
#         next(spamreader, None)
#         for row in spamreader:

#             identity = row[1]
#             fName = identity.replace(" ", "_")

#             if(LAST_IDENTITY != "" and not FOUND and LAST_IDENTITY != fName):
#                 continue
#             elif(LAST_IDENTITY != "" and LAST_IDENTITY == fName):
#                 FOUND = True
#                 continue

#             #create folder of identity, then download images using google api
#             current_path = os.path.join(CURRENT_FOLDER, NAME, fName)
#             if(not os.path.exists(current_path)):
#                 os.mkdir(current_path)

#             #get images from google api
#             print(identity)
        
#             query = identity

#             params = {
#                 "key": API_KEY,
#                 "cx": SEARCH_ENGINE_ID,
#                 "q": query,
#                 "searchType": "image",
#                 "imgType": "face",
#             }
#             count = 0
#             response = requests.get(URL, params=params)
#             for items in response.json()['items']:
#                 try:
#                     download_image(items['link'], os.path.join(current_path, str(count) + ".jpg"))
#                 except:
#                     print("Failed to download image")
#                 count += 1

#             response.close()
#             #UPDATE REQUESTS AND LIST IDENTITY
#             reqs += 1
#             if(reqs >= REQUESTS):
#                 print(f"!!!LAST IDENTITY: {identity}!!! save this value;")
#                 exit()

        
    

def fromFolder():
    reqs = 0
    for file in os.listdir(os.path.join(CURRENT_FOLDER, NAME)):
        path = os.path.join(CURRENT_FOLDER, NAME, file)
        if os.path.isdir(path):
            #make a request to google using start parameter
            identity = file.replace("_", " ")
            if(LAST_IDENTITY != "" and file < LAST_IDENTITY):
                continue
            if(LAST_IDENTITY != "" and file == LAST_IDENTITY):
                continue

            query = identity

            params = {
                "key": API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "q": query,
                "searchType": "image",
                "imgType": "face",
                "start": START
            }
            count = 0
            response = requests.get(URL, params=params)
            for items in response.json()['items']:
                download_image(items['link'], os.path.join(path, "real" ,str(name_offset + count) + ".jpg"))
                count += 1

            response.close()
            #UPDATE REQUESTS AND LIST IDENTITY
            reqs += 1
            if(reqs >= REQUESTS):
                print(f"!!!LAST IDENTITY: {identity}!!! save this value;")
                exit()


fromFolder()