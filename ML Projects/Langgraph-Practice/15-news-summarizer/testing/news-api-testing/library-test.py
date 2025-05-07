import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newsapi import NewsApiClient

print(os.getenv("NEWS_API_KEY"))
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
print(newsapi)

tech_sources = ['hacker-news', 'the-verge', 'techcrunch', 'wired']
# top_headlines = newsapi.get_top_headlines(
#                                       country='us',
#                                       category='business',
#                                       q='Finance and stock market')

top_headlines = newsapi.get_top_headlines(category='technology', country='us')

# Generate filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'tech_news_{timestamp}.json'
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# Save the response to a JSON file
with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(top_headlines, f, indent=4, ensure_ascii=False)

print(f"News data saved to: {filepath}")

# sources = newsapi.get_sources(category='technology', language='en', country='us')
# print(sources)

# for source in sources['sources']:
#     print(source['id'])
#     print(source['name'])
#     print(source['description'])
#     print(source['url'])