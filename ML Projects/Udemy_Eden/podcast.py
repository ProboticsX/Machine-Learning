from podcastfy.client import generate_podcast
from IPython.display import Audio, display

audio_file = generate_podcast(urls=["https://en.wikipedia.org/wiki/Podcast"])
