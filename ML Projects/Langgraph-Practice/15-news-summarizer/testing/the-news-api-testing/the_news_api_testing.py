import requests
import json

def fetch_news():
    # API endpoint
    url = "https://api.thenewsapi.com/v1/news/top"
    
    # Query parameters
    params = {
        "api_token": "nw3Qv4cBCZNQz6s4649e4IIvucceH4IAqyIegNyH",
        "categories": "tech",
        "published_on": "2025-06-09",
        "locale": "us",
        "limit": 3,
        "language": "en"
    }
    
    try:
        # Make the GET request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the JSON response
        data = response.json()
        
        # Extract only the required fields from each story
        formatted_stories = []
        for story in data["data"]:
            formatted_story = {
                "title": story["title"],
                "description": story["description"],
                "snippet": story["snippet"],
                "url": story["url"]
            }
            formatted_stories.append(formatted_story)
        
        # Return the formatted stories as JSON
        return json.dumps(formatted_stories, indent=2)
            
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Error fetching news: {str(e)}"})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Error parsing JSON response: {str(e)}"})

if __name__ == "__main__":
    result = fetch_news()
    print(result)
