import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz

def parse_date(date_str):
    """Parse date string from the news results"""
    try:
        # Remove UTC timezone info from string
        date_str = date_str.replace(", +0000 UTC", "")
        # Parse the date
        return datetime.strptime(date_str, "%m/%d/%Y, %I:%M %p").replace(tzinfo=pytz.UTC)
    except Exception as e:
        print(f"Error parsing date {date_str}: {e}")
        return None

def get_article_content(url):
    """Fetch and extract content from article URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text (remove excessive newlines and spaces)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        content = '\n'.join(lines)
        
        return content[:5000]  # Limit content length to 5000 characters
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return "Failed to fetch content"

def process_news():
    # Read the news results file
    with open('news_results.json', 'r') as f:
        news_data = json.load(f)
    
    # Get current time in UTC
    now = datetime.now(pytz.UTC)
    
    # Initialize results dictionary to group by position
    processed_articles = {}
    positions_processed = 0
    
    # Process each news result
    for news_item in news_data.get('news_results', []):
        position = news_item.get('position')
        current_group = {
            'position': position,
            'highlight': None,
            'related_stories': []
        }
        
        has_recent_content = False
        
        # Process highlight article
        highlight = news_item.get('highlight', {})
        date_str = highlight.get('date')
        
        if date_str:
            article_date = parse_date(date_str)
            
            if article_date and (now - article_date) <= timedelta(hours=24):
                highlight_article = {
                    'title': highlight.get('title'),
                    'source': highlight.get('source', {}).get('name'),
                    'authors': highlight.get('source', {}).get('authors', []),
                    'link': highlight.get('link'),
                    'published_date': date_str,
                    'thumbnail': highlight.get('thumbnail'),
                    'content': get_article_content(highlight.get('link'))
                }
                current_group['highlight'] = highlight_article
                has_recent_content = True
        
        # Process related stories
        for story in news_item.get('stories', []):
            date_str = story.get('date')
            
            if date_str:
                story_date = parse_date(date_str)
                
                if story_date and (now - story_date) <= timedelta(hours=24):
                    story_article = {
                        'title': story.get('title'),
                        'source': story.get('source', {}).get('name'),
                        'authors': story.get('source', {}).get('authors', []),
                        'link': story.get('link'),
                        'published_date': date_str,
                        'thumbnail': story.get('thumbnail'),
                        'content': get_article_content(story.get('link'))
                    }
                    current_group['related_stories'].append(story_article)
                    has_recent_content = True
        
        # Only add the group if it has content from the last 24 hours
        if has_recent_content:
            processed_articles[position] = current_group
            positions_processed += 1
            
            if positions_processed >= 10:
                break
    
    # Save processed articles to JSON file
    output = {
        'processed_time': now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        'total_positions': len(processed_articles),
        'news_groups': processed_articles
    }
    
    with open('processed_news.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(processed_articles)} news groups from the last 24 hours")

if __name__ == "__main__":
    process_news() 