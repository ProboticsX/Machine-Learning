from common_imports import *


@tool
def get_top_headlines(topic: str) -> str:
    """Gets the top headlines from the news API."""
    news_tool = NewsTool()
    result = news_tool.get_top_headlines(topic)
    return result

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
    return "Podcast created successfully"

@tool
def get_todays_date_and_day() -> str:
    """Gets the today's date and day."""
    today = datetime.now().strftime("%Y-%m-%d")
    day = datetime.now().strftime("%A")
    return f"{today} {day}"


# Get the project root directory (15-news-summarizer)
project_root = Path(__file__).parent.parent.parent
transcript_dir = project_root / "data" / "transcripts"
transcript_dir.mkdir(parents=True, exist_ok=True)
transcript_file = transcript_dir / "podcast-script.txt"

# Tools
top_headlines_agent_tools = [get_top_headlines]
podcast_transcript_writer_agent_tools = [write_to_file]
podcast_audio_generator_agent_tools = [create_podcast]
podcast_transcript_finetuner_agent_tools = [get_todays_date_and_day]