from common_imports import *

@tool
def get_top_headlines(category: str) -> str:
    """Gets the top headlines from the news API."""
    news_tool = NewsTool()
    result = news_tool.get_top_headlines(category)
    return result

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
def write_to_file(content: str) -> str:
    """Writes the content to a file in the data/transcripts folder."""
    with open(transcript_file, "w") as f:
        f.write(content)
    return f"File written successfully to {transcript_file}"

@tool
def create_podcast() -> str:
    """Creates a podcast from the transcript file."""
    audio_file = generate_podcast(
        transcript_file=str(transcript_file),
        tts_model="gemini",
        api_key_label="GEMINI_API_KEY"
    )
    print(f"Audio file generated and saved as: {audio_file}")
    return "Podcast created successfully and audio file path is: "+str(audio_file)

@tool
def write_to_summary_file(content: str) -> str:
    """Writes the content to a file in the data/summary folder."""
    with open(summary_file, "w") as f:
        f.write(content)
    return f"File written successfully to {summary_file}"

@tool
def write_summary_to_json_file(content: str) -> str:
    """Writes the content to a json file in the data/summary folder."""
    with open(summary_json_file, "w") as f:
        f.write(content)
    return f"File written successfully to {summary_json_file}"

@tool
def get_todays_date_and_day() -> str:
    """Gets the today's date and day."""
    today = datetime.now().strftime("%Y-%m-%d")
    day = datetime.now().strftime("%A")
    return f"{today} {day}"

@tool
def web_search(query: str) -> str:
    """Searches the web for the query."""
    search = TavilySearchResults()
    return search.invoke(query)

@tool
def push_headlines_to_firebase_db() -> str:
    """Pushes the headlines to the firebase database."""
    firebase_tools = FirebaseTools()
    result = firebase_tools.push_headlines_from_json(summary_json_file)
    return result

@tool
def push_audio_to_firebase_storage(audio_file_path: str) -> str:
    """Pushes the audio to the firebase storage."""
    try:
        firebase_tools = FireStorageAndFireStoreAudioTool(storage_bucket="iosapp-5d233.firebasestorage.app")
        result = firebase_tools.store_audio_file(audio_file_path)
        return str(result)
    except Exception as e:
        return f"Error pushing audio to Firebase Storage: {str(e)}"

# Get the project root directory (15-news-summarizer)
project_root = Path(__file__).parent.parent.parent
transcript_dir = project_root / "data" / "transcripts"
transcript_dir.mkdir(parents=True, exist_ok=True)
transcript_file = transcript_dir / "podcast_script.txt"
processsed_news_file_path = NewsTool().processed_news_path
summary_dir = project_root / "data" / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)
summary_file = summary_dir / "top_headlines_summary.txt"
summary_json_file = summary_dir / "top_headlines_summary.json"

# Tools
top_headlines_agent_tools = [get_top_headlines]
top_headlines_summarizer_tools = [read_json_file, web_search, write_to_summary_file, write_summary_to_json_file, push_headlines_to_firebase_db]

transcript_generator_agent_tools = [write_to_file]
podcast_audio_generator_agent_tools = [create_podcast, push_audio_to_firebase_storage]
podcast_transcript_critic_agent_tools = [get_todays_date_and_day]