from podcastfy.client import generate_podcast
from dotenv import load_dotenv

load_dotenv()

# audio_file = generate_podcast(
# 	urls=["https://www.gutenberg.org/cache/epub/20203/pg20203.txt"],
# 	llm_model_name="gpt-4o-mini",
# 	api_key_label="OPENAI_API_KEY"
# )

audio_file = generate_podcast(
	transcript_file="./data/transcripts/sample-transcript.txt",
	tts_model="gemini",
	api_key_label="GEMINI_API_KEY"
)

print(f"Audio file generated and saved as: {audio_file}")
