import os
from io import BytesIO
from typing import IO

from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

import subprocess
import time


load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)


def is_installed(name):
    try:
        subprocess.call([name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def play_stream(audio_stream):
    # Check if ffplay is installed, if not raise an error as it is necessary to stream audio
    player = "ffplay"
    if not is_installed(player):
        raise ValueError(f"{player} not found, necessary to stream audio.")

    # Define the command to stream audio using ffplay
    player_command = ["ffplay", "-autoexit", "-", "-nodisp"]

    # Start the player process with the command to stream audio
    player_process = subprocess.Popen(
        player_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write each chunk of audio data to the player process' stdin to play the audio
    for chunk in audio_stream:
        if chunk:
            player_process.stdin.write(chunk)
            player_process.stdin.flush() # Flush the buffer to ensure the data is written

    if player_process.stdin:
        player_process.stdin.close() # Close the stdin pipe to signal the end of the audio stream
    player_process.wait() # Wait for the player process to finish

# Define a function to convert text to speech and return a stream of audio data
def text_to_speech_stream(text: str) -> IO[bytes]:
    # Record the start time for performance measurement
    start_time = time.time()
    # Convert text to speech using the Eleven Labs API
    response = client.text_to_speech.convert(
        voice_id="iP95p4xoKVk53GoZ742B",  # Chris (pre-made voice)
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2", # "eleven_turbo_v2" is the fastest model
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    print("Streaming audio data...")

    # Create a BytesIO object to hold audio data
    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    # Return the stream for further use
    print("Audio data streaming complete.")
    end_time = time.time()
    elapsed_time = int((end_time - start_time) * 1000)
    print(f"Text-to-Speech ({elapsed_time}ms): {text}")

    return audio_stream


if __name__ == "__main__":
    audio_stream = text_to_speech_stream(
        "Hello, world! This is using the streaming API. I hope you enjoy it! Have a great day!"
    )
    play_stream(audio_stream)
