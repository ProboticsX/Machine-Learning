MODEL_NAME = "gpt-4.1-mini"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

MAX_TOP_HEADLINES_CRITIQUE_COUNT = 1
MAX_PODCAST_TRANSCRIPT_CRITIQUE_COUNT = 1 * 7
TOP_K_FOR_RAG_RETRIEVER = 3

NEWSPRESSO_SUPERVISOR_AGENT = "newspresso_supervisor_agent"

TOP_HEADLINES_SUPERVISOR_AGENT = "top_headlines_supervisor_agent"
TOP_HEADLINES_AGENT = "top_headlines_agent"
TOP_HEADLINES_IMAGE_AGENT = "top_headlines_image_agent"
TOP_HEADLINES_CLOUD_PUSHER_AGENT = "top_headlines_cloud_pusher_agent"

PODCAST_SUPERVISOR_AGENT = "podcast_supervisor_agent"
PODCAST_AUDIO_GENERATOR_AGENT = "podcast_audio_generator_agent"
PODCAST_TRANSCRIPT_SUPERVISOR_AGENT = "podcast_transcript_supervisor_agent"
PODCAST_TRANSCRIPT_GENERATOR_AGENT = "podcast_transcript_generator_agent"
PODCAST_TRANSCRIPT_CRITIC_AGENT = "podcast_transcript_critic_agent"

RAG_SUPERVISOR_AGENT = "rag_supervisor_agent"
RAG_RETRIEVER_AGENT = "rag_retriever_agent"
RAG_GRADER_AGENT = "rag_grader_agent"
RAG_GENERATOR_AGENT = "rag_generator_agent"