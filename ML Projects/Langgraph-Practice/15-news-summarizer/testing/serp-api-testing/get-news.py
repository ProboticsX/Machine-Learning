import os
import json
from serpapi import GoogleSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

params = {
    "api_key": os.getenv("SERPAPI_KEY"),
    "engine": "google_news",
    "gl": "us",
    "hl": "en",
    "topic_token": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB"
}

search = GoogleSearch(params)
results = search.get_dict()

# Save results to a JSON file
output_file = "news_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results have been saved to {output_file}")