from google.cloud import texttospeech
from dotenv import load_dotenv
from pathlib import Path
import os
import boto3
import logging
import json

# Load environment variables from the common .env file in the parent directory
load_dotenv(dotenv_path="../.env")

# Environment variable setup
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
digitalocean_access_key = os.getenv("DIGITALOCEAN_ACCESS_KEY")
digitalocean_secret_key = os.getenv("DIGITALOCEAN_SECRET_KEY")
digitalocean_endpoint = os.getenv("DIGITALOCEAN_SPACES_ENDPOINT")
digitalocean_bucket = os.getenv("DIGITALOCEAN_BUCKET")
audio_dir = os.getenv("AUDIO_DIR", "./ttsaudio")
log_file = os.getenv("LOG_FILE", "./log_tts.log")

# Google TTS default settings (can be overridden via API call)
default_language_code = os.getenv("TTS_LANGUAGE_CODE", "en-GB")
default_gender = os.getenv("TTS_SSML_GENDER", "FEMALE")
default_audio_encoding = os.getenv("TTS_AUDIO_ENCODING", "MP3")

# Set up logging
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Google Text-to-Speech client and DigitalOcean S3 client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials
s3_client = boto3.client(
    "s3",
    endpoint_url=digitalocean_endpoint,
    aws_access_key_id=digitalocean_access_key,
    aws_secret_access_key=digitalocean_secret_key
)

def check_pronunciation_exists(word):
    """Check if pronunciation file already exists locally."""
    sub_dir = Path(audio_dir) / word[:2]
    file_path = sub_dir / f"{word}.mp3"
    return file_path if file_path.exists() else None

def generate_pronunciation(word, language_code=None, gender=None):
    """Generate pronunciation for a word using Google TTS and save it as an MP3."""
    language_code = language_code or default_language_code
    gender = gender or default_gender
    logging.info(f"Generating pronunciation for word: {word} with language={language_code} and gender={gender}")

    # Initialize Google TTS client
    client = texttospeech.TextToSpeechClient()

    # TTS request configuration
    synthesis_input = texttospeech.SynthesisInput(text=word)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if gender == "FEMALE" else texttospeech.SsmlVoiceGender.MALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding[default_audio_encoding])

    # Perform TTS request
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # Create sub-directory based on the first two letters of the word
    sub_dir = Path(audio_dir) / word[:2]
    sub_dir.mkdir(parents=True, exist_ok=True)
    file_path = sub_dir / f"{word}.mp3"
    
    # Save audio content locally
    with open(file_path, "wb") as audio_file:
        audio_file.write(response.audio_content)
        logging.info(f"Pronunciation created and saved at {file_path}")

    # Upload to DigitalOcean Spaces
    try:
        s3_client.put_object(
            Bucket=digitalocean_bucket,
            Key=f"ttsaudio/{word[:2]}/{word}.mp3",
            Body=response.audio_content,
            ACL="public-read"
        )
        logging.info(f"Uploaded {word}.mp3 to DigitalOcean Spaces")
    except Exception as e:
        logging.error(f"Error uploading to DigitalOcean Spaces: {e}")

    return file_path

def process_sentence_pronunciation(sentence, language_code=None, gender=None):
    """Process each word in the sentence and check if pronunciation files exist or need to be generated."""
    words = sentence.split()
    results = {}

    for word in words:
        clean_word = word.lower().strip('.,!?')
        file_path = check_pronunciation_exists(clean_word)

        if file_path:
            # File already exists
            results[clean_word] = {
                "status": "exists",
                "path": str(file_path)
            }
        else:
            # Generate the file if it doesn't exist
            try:
                file_path = generate_pronunciation(clean_word, language_code, gender)
                results[clean_word] = {
                    "status": "generated",
                    "path": str(file_path)
                }
            except Exception as e:
                logging.error(f"Error generating pronunciation for '{clean_word}': {e}")
                results[clean_word] = {
                    "status": "error",
                    "message": str(e)
                }

    return {"success": True, "data": results}

# Example usage: Process a sentence
if __name__ == "__main__":
    sentence = "hello world"
    result = process_sentence_pronunciation(sentence)
    print(json.dumps(result, indent=2))

