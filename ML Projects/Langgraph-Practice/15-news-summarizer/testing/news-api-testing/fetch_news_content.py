import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from typing import Dict, List
import logging
import os
import sys
from dotenv import load_dotenv
from newsapi import NewsApiClient

# Set up logging with more detailed format and stream handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                            datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

def fetch_news() -> Dict:
    """Fetch latest news from NewsAPI."""
    try:
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        
        # Fetch top headlines
        logger.info("Fetching latest news from NewsAPI...")
        response = newsapi.get_top_headlines(
            language='en',
            country='us',
            page_size=100  # Get more articles to ensure we have enough with content
        )
        logger.info(f"Successfully fetched {len(response['articles'])} articles from NewsAPI")
        return response
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise

def fetch_article_content(url: str) -> str:
    """Fetch and extract the main content from a news article URL."""
    try:
        logger.info(f"Fetching content from URL: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        content_length = len(text)
        logger.info(f"Successfully extracted {content_length} characters of content")
        return text
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return ""

def process_articles(articles: List[Dict]) -> List[Dict]:
    """Process articles until we have 5 with full content or we've gone through all articles."""
    processed_articles = []
    article_index = 0
    
    while len(processed_articles) < 5 and article_index < len(articles):
        article = articles[article_index]
        logger.info(f"\nProcessing article {article_index + 1}/{len(articles)}")
        logger.info(f"Title: {article['title']}")
        logger.info(f"Source: {article['source']['name']}")
        
        content = fetch_article_content(article["url"])
        
        # Only add articles that have content
        if content.strip():
            article_data = {
                "source": article["source"]["name"],
                "author": article["author"],
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "published_at": article["publishedAt"],
                "content": content
            }
            processed_articles.append(article_data)
            logger.info(f"✓ Successfully processed article {len(processed_articles)}/5")
        else:
            logger.warning(f"✗ No content found for article: {article['title']}")
        
        article_index += 1
        if article_index < len(articles):
            logger.info("Waiting 1 second before processing next article...")
            time.sleep(1)  # Be nice to the servers
    
    logger.info(f"\nFinished processing. Successfully processed {len(processed_articles)} articles with full content")
    return processed_articles

def save_json(data: Dict, output_file: str):
    """Save data to a JSON file."""
    try:
        # Create news directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        # Fetch news articles
        news_data = fetch_news()
        
        # Save raw news data
        save_json(news_data, "news/raw_news.json")
        
        # Process articles
        processed_articles = process_articles(news_data["articles"])
        
        # Save processed articles
        output_data = {
            "processed_at": datetime.utcnow().isoformat(),
            "total_articles": len(processed_articles),
            "articles": processed_articles
        }
        save_json(output_data, "news/processed_news.json")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 