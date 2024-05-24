import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()


class TimingManager:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def total_time(self):
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return None


timing_manager = TimingManager()


class LanguageModelProcessor:
    """
    The LanguageModelProcessor class processes user input text and generates a response using a language model.

    This class uses the ChatGroq instance to interact with the language model and generate a response.
    The class also uses the ConversationBufferMemory instance to store the conversation history and the LLMChain instance to
    manage the conversation flow.
    """

    def __init__(self):
        """
        Initializes an instance of the LanguageModelProcessor class.

        The __init__ method sets up the necessary components for the language model processor,
        including the ChatGroq instance, ConversationBufferMemory instance, system prompt,
        and the LLMChain instance.

        Parameters:
        - None

        Returns:
        - None
        """
        # Initialize the language model with ChatGroq using the LLaMA 3 8B model
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        # Initialize the conversation memory for storing chat history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Fetch the system prompt from the system_prompt.txt file
        with open("src/system-prompt.txt", "r") as file:
            system_prompt = file.read().strip()

        # Create the chat prompt template with the system prompt, chat history, and human message
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )

        # Create the conversation chain with the language model and memory
        self.conversation = LLMChain(
            llm=self.llm, prompt=self.prompt, memory=self.memory
        )

    def process(self, text):
        """
        Processes the input text and generates a response using the language model.

        The process method adds the user message to the conversation memory, invokes the language model
        to generate a response, adds the AI response to the conversation memory, and returns the generated response.

        Parameters:
        - text (str): The user input text.

        Returns:
        - str: The generated response from the language model.
        """
        # Adds the user message to the conversation memory for chat history
        self.memory.chat_memory.add_user_message(text)

        # Record the time before sending the request
        start_time = time.time()

        # Generate a response from the LLaMA 3 8B language model
        response = self.conversation.invoke({"text": text})
        end_time = time.time()  # Record the time after receiving the response

        # Adds the AI response to the conversation memory for chat history
        self.memory.chat_memory.add_ai_message(response["text"])

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"\nLLM Response (took {elapsed_time}ms):\n{response['text']}")
        return response["text"]


class TextToSpeech:
    def __init__(self, timing_manager):
        self.timing_manager = timing_manager

    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(
            DEEPGRAM_URL, stream=True, headers=headers, json=payload
        ) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if (
                        first_byte_time is None
                    ):  # Check if this is the first chunk received
                        first_byte_time = (
                            time.time()
                        )  # Record the time when the first byte is received
                        ttfb = int(
                            (first_byte_time - start_time) * 1000
                        )  # Calculate the time to first byte
                        self.timing_manager.end()
                        print(f"\nTTS Time to First Byte (TTFB): {ttfb}ms")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()


class TranscriptAggregator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return " ".join(self.transcript_parts)


transcript_collector = TranscriptAggregator()


async def get_transcript(callback):
    """
    Retrieves the transcript of a speech input using the Deepgram API.

    Args:
        callback (function): A callback function to handle the retrieved transcript.

    Returns:
        None
    """
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")

        print("\n=========================================")
        print("Listening for user input now...")
        print("=========================================\n\n")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript

            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    # start_time = time.time()
                    print(f"User Transcript: \n{full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # Set the options for the live transcription
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,  # Waits for 300ms of silence before ending the transcription
            smart_format=True,
        )

        # Start the live transcription connection
        await dg_connection.start(options)

        # Start the microphone input using the default system microphone
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely
        # start_time = time.time()
        timing_manager.start()

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        # end_time = time.time()
        # print(f"Transcription Time: {end_time - start_time:.2f} seconds")
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):

        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech(timing_manager)
            tts.speak(llm_response)

            # calculating total time
            total_time = timing_manager.total_time()

            if total_time is not None:
                print("Total round-trip time (RTT):", total_time, "\n")
            else:
                print("Timing data is incomplete.")

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""


if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
