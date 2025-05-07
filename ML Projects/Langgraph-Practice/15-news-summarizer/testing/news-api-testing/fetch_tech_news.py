import sys
import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from typing import Dict, List
import logging
from dotenv import load_dotenv
from newsapi import NewsApiClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_news() -> Dict:
    """Fetch latest news from NewsAPI."""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("NEWS_API_KEY")
        
        if not api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
        
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=api_key)
        
        # Fetch top headlines
        logger.info("Fetching latest news...")
        top_headlines = newsapi.get_top_headlines(country='us')
        
        # Save raw response to a JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_filename = f'news_{timestamp}.json'
        raw_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), raw_filename)
        
        with open(raw_filepath, 'w', encoding='utf-8') as f:
            json.dump(top_headlines, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Raw news data saved to: {raw_filepath}")
        return top_headlines
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise

def fetch_article_content(url: str) -> str:
    """Fetch and extract the main content from a news article URL."""
    try:
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
        
        return text
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return ""

def process_articles(articles: List[Dict]) -> List[Dict]:
    """Process articles until we have 10 with full content or we've gone through all articles."""
    processed_articles = []
    article_index = 0
    
    while len(processed_articles) < 10 and article_index < len(articles):
        article = articles[article_index]
        logger.info(f"Processing article {article_index + 1}/{len(articles)}: {article['title']}")
        
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
            logger.info(f"Successfully processed article {len(processed_articles)}/10")
        else:
            logger.warning(f"No content found for article: {article['title']}")
        
        article_index += 1
        time.sleep(1)  # Be nice to the servers
    
    logger.info(f"Processed {len(processed_articles)} articles with full content")
    return processed_articles

def save_processed_articles(articles: List[Dict], output_file: str):
    """Save the processed articles to a JSON file."""
    output_data = {
        "processed_at": datetime.utcnow().isoformat(),
        "total_articles": len(articles),
        "articles": articles
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved processed articles to {output_file}")
    except Exception as e:
        logger.error(f"Error saving processed articles: {e}")
        raise

def main():
    try:
        # Fetch latest news
        news_data = fetch_news()
        
        # Process articles
        processed_articles = process_articles(news_data["articles"])
        
        # Save processed articles
        output_file = f"processed_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_processed_articles(processed_articles, output_file)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 