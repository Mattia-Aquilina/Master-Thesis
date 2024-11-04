import requests
import os

# Bearer Token for OAuth 2.0 Bearer Token Authentication
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAP8xwQEAAAAAPq8HjKE%2FhP0h3pnWWtGwBWa1k8Y%3DZbsFWyXogDUssP6Y5pQfa1yl3yxWBdmoMjdPW0AZJrBRqCw9Vq'

# Define the search query - in this case, we're looking for tweets with images
search_query = 'has:images -is:retweet'

# Set up the request headers including the bearer token
headers = {
    'Authorization': f'Bearer {BEARER_TOKEN}',
}

# Twitter API v2 search endpoint URL
url = 'https://api.twitter.com/2/tweets/search/all'

# Parameters including media fields to get the media (images) attached
params = {
    'query': search_query,  # Search for tweets with images
    'tweet.fields': 'created_at',  # Additional fields like time of tweet
    'expansions': 'attachments.media_keys',  # Get media attachments
    'media.fields': 'url',  # Get the media URL
    'max_results': 10,  # Number of tweets to retrieve
}

# Make the GET request to Twitter API
response = requests.get(url, headers=headers, params=params)

# If request is successful
if response.status_code == 200:
    json_response = response.json()

    # Extract the media information if available
    media_info = json_response.get('includes', {}).get('media', [])

    for media in media_info:
        if media['type'] == 'photo':
            print(f"Image URL: {media['url']}")

        # Optionally, you can download the image
        image_url = media['url']
        image_data = requests.get(image_url).content
        with open(os.path.join('images', f"{media['media_key']}.jpg"), 'wb') as img_file:
            img_file.write(image_data)
else:
    print(f"Failed to retrieve data: {response.json()}")
