from common_imports import *
import re

@tool
def read_json_file(file_path: str) -> str:
    """Reads and returns the contents of a JSON file.
    
    Args:
        file_path: Path to the JSON file to read
        
    Returns:
        str: Contents of the JSON file as a formatted string
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in file {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_text_file(content: str, file_path: str) -> str:
    """Writes the content to a file in the data/transcripts folder."""
    with open(file_path, "w") as f:
        f.write(content)
    return f"File written successfully to {file_path}"

@tool
def create_podcast(file_path: str) -> str:
    """Creates a podcast from the transcript file."""
    audio_file = generate_podcast(
        transcript_file=str(file_path),
        tts_model="gemini",
        api_key_label="GEMINI_API_KEY"
    )
    print(f"Audio file generated and saved as: {audio_file}")
    return "Podcast created successfully and audio file path is: "+str(audio_file)

@tool
def write_json_file(content: str, file_path: str) -> str:
    """Writes the content to a json file in the data/summary folder."""
    with open(file_path, "w") as f:
        f.write(content)
    return f"File written successfully to {file_path}"

@tool
def get_todays_date_and_day() -> str:
    """
    Gets the today's date and day.
    Returns:
        str: The today's date and day in the format of YYYY-MM-DD and the day of the week.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    day = datetime.now().strftime("%A")
    return f"{today} {day}"

@tool
def web_search(query: str) -> str:
    """Searches the web for the query."""
    search = TavilySearchResults()
    return search.invoke(query)

@tool
def push_headlines_to_firebase_db(category: str, file_path: str) -> str:
    """Pushes the headlines to the firebase database."""
    firebase_tools = FirebaseTools()
    result = firebase_tools.push_headlines_from_json(file_path, category=category)
    return result

@tool
def push_audio_to_firebase_storage(audio_file_path: str, category: str) -> str:
    """Pushes the audio to the firebase storage."""
    try:
        firebase_tools = FireStorageAndFireStoreAudioTool(storage_bucket="iosapp-5d233.firebasestorage.app")
        result = firebase_tools.store_audio_file(audio_file_path, category=category)
        return str(result)
    except Exception as e:
        return f"Error pushing audio to Firebase Storage: {str(e)}"

@tool
def validate_json_file() -> str:
    """Validates the summary JSON file for common issues and returns validation status.
    
    Checks for:
    - File existence
    - Valid JSON syntax
    - Control characters
    - Invalid characters
    - Unclosed brackets/braces
    - Proper UTF-8 encoding
    
    Returns:
        str: Validation result with details if any issues are found
    """
    try:
        # Check if file exists
        if not os.path.exists(summary_json_file):
            return "❌ Validation failed: File not found"
        
        # Read file content
        with open(summary_json_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for control characters
        control_chars = ''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))
        control_char_re = re.compile('[%s]' % re.escape(control_chars))
        if control_char_re.search(content):
            return "❌ Validation failed: File contains control characters"
        
        # Check for unclosed brackets/braces
        if content.count('{') != content.count('}'):
            return "❌ Validation failed: Unmatched curly braces"
        if content.count('[') != content.count(']'):
            return "❌ Validation failed: Unmatched square brackets"
        
        # Try to parse JSON
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return f"❌ Validation failed: Invalid JSON syntax - {str(e)}"
        
        # Check for invalid characters
        try:
            content.encode('ascii')
        except UnicodeEncodeError:
            # If we can't encode as ASCII, check if it's valid UTF-8
            try:
                content.encode('utf-8')
            except UnicodeEncodeError:
                return "❌ Validation failed: File contains invalid characters"
        
        return "✅ Validation passed: File is a valid JSON"
        
    except Exception as e:
        return f"❌ Validation failed: Unexpected error - {str(e)}"

def get_perplexity_payload(question: str, return_images: bool = False):
    payload = {
        "model": perplexity_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions and provides information."},
            {"role": "user", "content": question}
        ],
        "return_images": return_images,
    }
    return payload

def get_perplexity_headers():
    headers = {
        "Authorization": f"Bearer {os.getenv("PERPLEXITY_API_KEY")}",
        "Content-Type": "application/json"
    }
    return headers

@tool
def get_perplexity_response(question: str, return_images: bool = False):
    """Gets the response from the perplexity API.
    Args:
        question: The question to ask the perplexity API.
        return_images: Whether to return images in the response. If True, the response will contain the image URLs.
    Returns:
        str: The response from the perplexity API.
    """

    payload = get_perplexity_payload(question, return_images)
    headers = get_perplexity_headers()
    response = requests.request("POST", PERPLEXITY_API_URL, json=payload, headers=headers)
    return response.json()

@tool
def get_perplexity_response_with_image(headline_content: str) -> str:
    """Gets the image URL from the perplexity API.
    Args:
        headline_content: The content of the headline related to the image URL.
    Returns:
        str: The image URL from the perplexity API.
    """
    payload = get_perplexity_payload(headline_content, return_images=True)
    headers = get_perplexity_headers()
    response = requests.request("POST", PERPLEXITY_API_URL, json=payload, headers=headers)
    return response.json()

@tool
def check_image_url(image_url: str) -> str:
    """Checks if the image URL is valid and points to an actual image file.
    Args:
        image_url: The image URL to check.
    Returns:
        str: The image URL if it is valid, otherwise returns an error message.
    """
    try:
        # Check if URL contains a valid image extension anywhere
        valid_image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        if not any(ext in image_url.lower() for ext in valid_image_extensions):
            return "❌ URL does not point to a valid image file (missing image extension)"

        # Make HEAD request first to check content type without downloading
        head_response = requests.head(image_url, allow_redirects=True)
        if head_response.status_code != 200:
            return "❌ Image URL is not valid or the image cannot be accessed."

        # Check content type
        content_type = head_response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            return f"❌ URL does not point to an image (content-type: {content_type})"

        # Make GET request to verify image can be opened
        response = requests.get(image_url)
        if response.status_code == 200:
            return "✅ Image URL is valid and points to an actual image file."
        else:
            return "❌ Image URL is not valid or the image cannot be opened."
    except Exception as e:
        return f"❌ Error checking image URL: {str(e)}"

project_root = Path(__file__).parent.parent.parent

transcript_dir = project_root / "data" / "transcripts"
transcript_dir.mkdir(parents=True, exist_ok=True)
podcast_transcript_file = transcript_dir / "podcast_transcript.txt"

summary_dir = project_root / "data" / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)
summary_json_file = summary_dir / "top_headlines_summary.json"

# Tools
category_extractor_agent_tools = [get_todays_date_and_day]
top_headlines_agent_tools = [get_perplexity_response, write_json_file, validate_json_file]
top_headlines_image_agent_tools = [get_perplexity_response_with_image, check_image_url, read_json_file, write_json_file, validate_json_file]
top_headlines_firebase_pusher_agent_tools = [push_headlines_to_firebase_db, read_json_file, write_json_file, validate_json_file]

transcript_generator_agent_tools = [write_text_file, read_json_file]
podcast_audio_generator_agent_tools = [create_podcast, push_audio_to_firebase_storage]