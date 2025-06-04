import json
import os

def extract_headlines():
    # Read the JSON file from the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'top_headlines_summary.json')
    
    with open(json_path, 'r') as file:
        headlines = json.load(file)
    
    # Create output text file in the same directory
    output_path = os.path.join(script_dir, 'formatted_headlines.txt')
    with open(output_path, 'w') as file:
        for headline in headlines:
            # Write each headline in the specified format
            file.write(f"- Title: {headline['title']}\n")
            file.write(f"    Description: {headline['description']}\n")
            file.write(f"    Content_Summary: {headline['content_summary']}\n\n")

if __name__ == "__main__":
    extract_headlines() 