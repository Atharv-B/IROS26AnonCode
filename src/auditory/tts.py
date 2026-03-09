"""
A text to speech module that can be used for auditory feedback.
"""
from yapper import Yapper
import threading

class TextToSpeech:
    def __init__(self):
        # Initialize the TTS engine here
        self.tts = Yapper()

    def speak_sync(self, text: str):
        # Convert text to speech and play it (synchonous, blocking)
        print(f"Speaking: {text}")  # Placeholder for actual TTS functionality
        self.tts.yap(text)

    def speak_async(self, text: str):
        # Convert text to speech and play it asynchronously
        # BE CAREFUL: multiple async calls will result in overlapping audio!
        print(f"Speaking asynchronously: {text}")
        threading.Thread(target=self.tts.yap, args=(text,)).start()

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak_sync("I think I know what you want to do! Let me help you with that.")
    for i in range(10):
        print(i)

    tts.speak_async("Looks like you are trying to pick up the water bottle.")
    for i in range(10):
        print(i)