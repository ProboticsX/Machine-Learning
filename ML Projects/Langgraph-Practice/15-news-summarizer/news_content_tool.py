import subprocess
import json
from pathlib import Path
from typing import Dict
import logging
import os
import sys

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

class NewsContentTool:
    def __init__(self, script_dir: str = None):
        """
        Initialize the NewsContentTool.
        
        Args:
            script_dir (str, optional): Directory containing fetch_news_content.py. 
                                      If None, uses the testing/news-api-testing directory.
        """
        if script_dir is None:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent
            self.script_dir = current_dir / "testing" / "news-api-testing"
        else:
            self.script_dir = Path(script_dir)
            
        self.script_path = self.script_dir / "fetch_news_content.py"

    def get_top_headlines(self) -> Dict:
        """
        Run fetch_news_content.py to fetch and process news articles.
        
        Returns:
            Dict: Processed articles with full content in the following format:
            {
                "processed_at": "timestamp",
                "total_articles": int,
                "articles": [
                    {
                        "source": str,
                        "author": str,
                        "title": str,
                        "description": str,
                        "url": str,
                        "published_at": str,
                        "content": str
                    },
                    ...
                ]
            }
        """
        try:
            # Run the script
            logger.info("Running fetch_news_content.py")
            
            # Run the script with real-time output
            process = subprocess.Popen(
                ["python", str(self.script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.script_dir
            )

            # Stream the output in real-time
            for line in process.stdout:
                print(line, end='', flush=True)

            # Wait for the process to complete
            return_code = process.wait()

            # Check if script ran successfully
            if return_code != 0:
                logger.error("Script execution failed")
                raise RuntimeError("Script execution failed")

            # Read the output file
            output_file = self.script_dir / "news" / "processed_news.json"
            if not output_file.exists():
                raise FileNotFoundError("Output file not found after script execution")
            
            # Read and return the processed articles
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error in get_top_headlines: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Create an instance of the tool
    news_tool = NewsContentTool()
    
    # Process news articles
    result = news_tool.get_top_headlines()
    
    # Print the number of processed articles
    print(f"Processed {result['total_articles']} articles") 