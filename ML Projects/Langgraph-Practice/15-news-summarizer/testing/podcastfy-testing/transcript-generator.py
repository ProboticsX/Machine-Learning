from podcastfy.client import generate_podcast
from dotenv import load_dotenv

load_dotenv()

transcript_file = generate_podcast(
	urls=["https://www.gutenberg.org/cache/epub/20203/pg20203.txt"],
	transcript_only=True,
	llm_model_name="gpt-4o-mini",
	api_key_label="OPENAI_API_KEY"
)

print(f"Transcript generated and saved as: {transcript_file}")
# Read and print the first 20 characters from the transcript file
with open(transcript_file, 'r') as file:
	transcript_content = file.read(100)
	print(f"First 100 characters of the transcript: {transcript_content}")