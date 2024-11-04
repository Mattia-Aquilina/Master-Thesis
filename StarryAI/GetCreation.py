import requests

url = "https://api.starryai.com/creations/?limit=50&offset=0"

headers = {"accept": "application/json","X-API-Key": "xjO_36phyS6ZD2SOovLssn9ryGmsBg"}

response = requests.get(url, headers=headers)

for item in response.json():
    print(item['images'][0]['url'])