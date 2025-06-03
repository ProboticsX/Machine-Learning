import base64
import mimetypes
import os
import struct
from typing import Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GeminiMultiSpeakerTool:
    """Tool for generating podcast audio using Gemini's speech capabilities."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        self.model = "gemini-2.5-flash-preview-tts"
    
    def _save_binary_file(self, file_name: str, data: bytes) -> str:
        """Save binary data to a file.
        
        Args:
            file_name: Name of the file to save
            data: Binary data to save
            
        Returns:
            Path to the saved file
        """
        with open(file_name, "wb") as f:
            f.write(data)
        return file_name

    def _convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Convert audio data to WAV format.
        
        Args:
            audio_data: Raw audio data
            mime_type: MIME type of the audio data
            
        Returns:
            WAV formatted audio data
        """
        parameters = self._parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            chunk_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size
        )
        return header + audio_data

    def _parse_audio_mime_type(self, mime_type: str) -> Dict[str, int]:
        """Parse audio MIME type parameters.
        
        Args:
            mime_type: Audio MIME type string
            
        Returns:
            Dictionary containing bits_per_sample and rate
        """
        bits_per_sample = 16
        rate = 24000

        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass

        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def generate_podcast(self, raw_text: str, output_path: str = "podcast_audio") -> Dict[str, Any]:
        """Generate podcast audio from text.
        
        Args:
            raw_text: Text to convert to speech
            output_path: Path to save the generated audio file
            
        Returns:
            Dictionary containing the path to the generated audio file
        """
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="""Read aloud in a warm, welcoming tone. You are hosting a multi spekaer news podcast so make it more engaging and interesting to listen to."""+raw_text),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 1",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Zephyr"
                                )
                            ),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 2",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Puck"
                                )
                            ),
                        ),
                    ]
                ),
            ),
        )

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
                
            if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                
                if file_extension is None:
                    file_extension = ".wav"
                    data_buffer = self._convert_to_wav(inline_data.data, inline_data.mime_type)
                
                output_file = f"{output_path}{file_extension}"
                self._save_binary_file(output_file, data_buffer)
                return {"audio_file": output_file}
            
        return {"error": "Failed to generate audio"}

if __name__ == "__main__":
    gemini_multispeaker_tool = GeminiMultiSpeakerTool()
    gemini_multispeaker_tool.generate_podcast("Speaker 1: This is a test podcast. Speaker 2: This is a test podcast.")