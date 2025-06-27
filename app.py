# app.py
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from murf import Murf # Import the Murf SDK
import io # Used for handling audio in memory
import requests # Still needed to download the audio from Murf's URL
import json # Used for handling JSON responses from LLM API
import base64 # For encoding/decoding images
import os

app = Flask(__name__, static_folder='.', static_url_path='') # Serve static files from current directory
CORS(app) # Enable CORS for all routes

# --- API Key Configuration ---
# IMPORTANT: If you are running this Flask app LOCALLY (e.g., via 'python app.py' in your terminal),
# you MUST replace the empty strings below with your actual API keys.
# If you are running this within a Canvas preview environment, leave them as empty strings.

# Murf AI API Key (for Text-to-Speech)
# Replace "YOUR_MURF_AI_API_KEY" with your actual Murf AI API Key.
MURF_API_KEY = os.environ.get("MURF_API_KEY", "YOUR_MURF_AI_API_KEY_FALLBACK") 

# Gemini API Key (for Text Humanization and Book Writing)
# Get your Gemini API key from Google AI Studio (https://aistudio.google.com/app/apikey)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# --- Initialize API Clients/URLs ---
# Initialize the Murf client
client = Murf(api_key=MURF_API_KEY)

# Construct API URLs using the (potentially filled) API keys
GEMINI_TEXT_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Pollinations.AI URL for Book Cover Generation - No API Key needed for this service
POLLINATIONS_IMAGE_API_BASE_URL = "https://image.pollinations.ai/prompt/"

# --- Debugging: Print API Keys at startup ---
print(f"DEBUG: MURF_API_KEY (first 5 chars): {MURF_API_KEY[:5]}...")
print(f"DEBUG: GEMINI_API_KEY (first 5 chars): {GEMINI_API_KEY[:5]}...")
# IMAGEN_API_KEY is no longer used, so we remove its debug print


@app.route('/')
def serve_index():
    """Serves the index.html file."""
    return send_file('index.html')

@app.route('/voices', methods=['GET'])
def get_voices():
    """
    Retrieves a list of available Murf AI voices.
    We are using a hardcoded list for simplicity.
    IMPORTANT: Please refer to Murf AI's official Voice Library
    to find the exact 'voice_id's that are available and
    active for your account, and update this list accordingly.
    """
    voices_list = [
        # Ensure these voice IDs are valid and available on your Murf AI plan.
        # Example valid US English voices:
        {"name": "Natalie (US English, Female, Neural)", "id": "en-US-natalie", "gender": "Female", "locale": "en-US", "voice_type": "Neural"},
        {"name": "Noah (US English, Male, Neural)", "id": "en-US-noah", "gender": "Male", "locale": "en-US", "voice_type": "Neural"},
        {"name": "Jenny (US English, Female, Neural)", "id": "en-US-jenny", "gender": "Female", "locale": "en-US", "voice_type": "Neural"},
        {"name": "Terrell (US English, Male, Neural)", "id": "en-US-terrell", "gender": "Male", "locale": "en-US", "voice_type": "Neural"},
        {"name": "Samantha (US English, Female, Neural)", "id": "en-US-samantha", "gender": "Female", "locale": "en-US", "voice_type": "Neural"},
        {"name": "John (US English, Male, Neural)", "id": "en-US-john", "gender": "Male", "locale": "en-US", "voice_type": "Neural"},
        {"name": "Olivia (British English, Female, Neural)", "id": "en-GB-olivia", "gender": "Female", "locale": "en-GB", "voice_type": "Neural"},
        {"name": "Liam (Australian English, Male, Neural)", "id": "en-AU-liam", "gender": "Male", "locale": "en-AU", "voice_type": "Neural"}, 
        # Indian English Voices from the provided image:
        {"name": "Aarav (Indian English, Male, Conversational)", "id": "en-IN-aarav", "gender": "Male", "locale": "en-IN", "voice_type": "Conversational"},
        {"name": "Arohi (Indian English, Female, Conversational)", "id": "en-IN-arohi", "gender": "Female", "locale": "en-IN", "voice_type": "Conversational"},
        {"name": "Rohan (Indian English, Male, Conversational)", "id": "en-IN-rohan", "gender": "Male", "locale": "en-IN", "voice_type": "Conversational"},
        {"name": "Alia (Indian English, Female, Promo)", "id": "en-IN-alia", "gender": "Female", "locale": "en-IN", "voice_type": "Promo"},
        {"name": "Surya (Indian English, Male, Documentary)", "id": "en-IN-surya", "gender": "Male", "locale": "en-IN", "voice_type": "Documentary"},
        {"name": "Priya (Indian English, Female, Conversational)", "id": "en-IN-priya", "gender": "Female", "locale": "en-IN", "voice_type": "Conversational"},
        {"name": "Shivani (Indian English, Female, Conversational)", "id": "en-IN-shivani", "gender": "Female", "locale": "en-IN", "voice_type": "Conversational"},
        {"name": "Isha (Indian English, Female, Conversational)", "id": "en-IN-isha", "gender": "Female", "locale": "en-IN", "voice_type": "Conversational"},
        {"name": "Eashwar (Indian English, Male, Conversational)", "id": "en-IN-eashwar", "gender": "Male", "locale": "en-IN", "voice_type": "Conversational"},
        # Add more Murf voices as needed by looking up their voice_id in Murf Studio or docs
    ]
    
    # Sort voices for better display in the frontend
    voices_list.sort(key=lambda v: (v['locale'], v['name']))
    return jsonify(voices_list)

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """
    Synthesizes text into speech using Murf AI.
    Expects JSON payload with 'text', 'voiceId', 'pitch', 'speed', and 'action'.
    Returns:
        Audio content (MP3) or JSON error.
    """
    try:
        data = request.json
        text = data.get('text')
        voice_id = data.get('voiceId') # Use 'id' from our custom voices list for Murf
        pitch_val = float(data.get('pitch', 0.0)) # Pitch value from frontend (-20 to 20)
        speed_val = float(data.get('speed', 1.0)) # Speed value from frontend (0.25 to 4.0)
        action = data.get('action') # 'play' or 'download'

        if not text or not voice_id:
            return jsonify({"error": "Missing text or voice ID"}), 400

        # Murf's pitch and rate parameters:
        # pitch: Integer between -50 and 50. Our frontend is -20 to 20.
        murf_pitch = int(pitch_val) 

        # rate: Integer between -50 and 50. Our frontend is 0.25 to 4.0.
        if speed_val < 1.0: # Slower speeds (0.25 to 0.95) mapped to -50 to -1
            murf_rate = int(((speed_val - 0.25) / 0.75) * 50 - 50) 
        elif speed_val > 1.0: # Faster speeds (1.05 to 4.0) mapped to 1 to 50
            murf_rate = int(((speed_val - 1.0) / 3.0) * 50) 
        else: # Normal speed (1.0) mapped to 0
            murf_rate = 0
        
        # Ensure rate stays within Murf's expected range [-50, 50]
        murf_rate = max(-50, min(50, murf_rate))


        # Call Murf AI API using the SDK
        response = client.text_to_speech.generate(
            text=text,
            voice_id=voice_id,
            format="MP3", # Murf supports MP3
            rate=murf_rate,
            pitch=murf_pitch
        )
        
        # The Murf SDK returns a URL to the generated audio file
        audio_url = response.audio_file

        if not audio_url:
            raise Exception("Murf AI did not return an audio file URL.")

        # Download the audio content from the URL
        audio_response = requests.get(audio_url, stream=True)
        audio_response.raise_for_status() # Raise an exception for HTTP errors during download

        audio_buffer = io.BytesIO(audio_response.content)

        if action == 'download':
            return send_file(
                audio_buffer,
                mimetype='audio/mpeg',
                as_attachment=True,
                download_name='converted_audio.mp3'
            )
        else: # Default to 'play' action
            return audio_buffer.getvalue(), 200, {'Content-Type': 'audio/mpeg'}

    except Exception as e:
        print(f"Error during Murf AI speech synthesis: {e}")
        return jsonify({"error": "Speech synthesis failed with Murf AI", "details": str(e)}), 500

@app.route('/humanize_text', methods=['POST'])
def humanize_text():
    """
    Takes text from the user and humanizes it using the Gemini API.
    """
    try:
        data = request.json
        text_to_humanize = data.get('text')

        if not text_to_humanize:
            return jsonify({"error": "No text provided for humanization"}), 400

        prompt = f"Humanize the following text, making it sound 100% natural and written by a human. Ensure it is grammatically correct, flows well, and uses appropriate tone and vocabulary. Do not add any introductory or concluding remarks, just the humanized text:\n\n{text_to_humanize}"
        
        chat_history = []
        chat_history.append({ "role": "user", "parts": [{ "text": prompt }] }) 
        
        payload = { "contents": chat_history }

        # Make the fetch call to the Gemini API
        gemini_response = requests.post(
            GEMINI_TEXT_API_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        gemini_response.raise_for_status() # Raise an exception for HTTP errors
        
        # Debugging: Print raw response from Gemini API
        print(f"DEBUG: Gemini Humanize Raw Response: {gemini_response.text}")

        result = gemini_response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            humanized_text = result['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"humanizedText": humanized_text})
        else:
            raise Exception("Gemini API response format unexpected or missing content.")

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API for humanization: {e}")
        # Debugging: Print response content if available for request errors
        if e.response is not None:
            print(f"DEBUG: Gemini Humanize Error Response Content: {e.response.text}")
        return jsonify({"error": "Failed to humanize text via Gemini API", "details": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error during text humanization: {e}")
        return jsonify({"error": "An unexpected error occurred during humanization", "details": str(e)}), 500

@app.route('/generate_book', methods=['POST'])
def generate_book():
    """
    Generates a book with chapters based on topic/title, type, and word/chapter limits.
    This uses an iterative approach as single large requests are not feasible.
    """
    try:
        data = request.json
        topic = data.get('topic')
        word_limit = int(data.get('word_limit', 5000)) # Default to 5000 words if not specified
        num_chapters = int(data.get('num_chapters', 5)) # New: Number of chapters
        book_type = data.get('book_type', 'novel') # New: Type of book

        if not topic:
            return jsonify({"error": "Book topic or title is required"}), 400
        
        if word_limit > 50000: # Enforce the 50,000 word limit
            word_limit = 50000
            print("Word limit adjusted to max 50,000 words.")
        
        if num_chapters <= 0:
            return jsonify({"error": "Number of chapters must be at least 1"}), 400

        book_content = []
        current_word_count = 0
        
        # --- Step 1: Generate Book Outline/Chapters ---
        outline_prompt = (
            f"Please generate a detailed outline for a {book_type} book titled/on the topic: '{topic}'. "
            f"Include a compelling title, a brief synopsis, and exactly {num_chapters} chapter titles with a 1-2 sentence summary for each chapter. "
            "Format the output as:\n\nTitle: [Book Title]\nSynopsis: [Synopsis]\nChapters:\n1. [Chapter 1 Title]: [Summary]\n2. [Chapter 2 Title]: [Summary]\n..."
        )
        chat_history = [{"role": "user", "parts": [{"text": outline_prompt}]}]
        payload = {"contents": chat_history}
        
        outline_response = requests.post(
            GEMINI_TEXT_API_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        outline_response.raise_for_status()
        
        # Debugging: Print raw response from Gemini API
        print(f"DEBUG: Gemini Book Outline Raw Response: {outline_response.text}")

        outline_result = outline_response.json()
        outline_text = ""
        if outline_result.get('candidates') and outline_result['candidates'][0].get('content') and outline_result['candidates'][0]['content'].get('parts'):
            outline_text = outline_result['candidates'][0]['content']['parts'][0]['text']
            book_content.append(f"# {topic}\n\n## Outline\n{outline_text}\n\n")
            current_word_count += len(outline_text.split())
        else:
            raise Exception("Gemini API response for outline format unexpected or missing content.")

        # --- Parse Chapters from Outline ---
        # More robust parsing for chapter titles assuming numbering (1., 2., etc.)
        chapters = []
        lines = outline_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(tuple(f"{i}." for i in range(1, num_chapters + 1))):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    chapter_title = parts[0].strip()
                    chapters.append(chapter_title)
        
        # Fallback if parsing fails or fewer chapters are parsed than requested
        if len(chapters) != num_chapters:
            print(f"Warning: Parsed {len(chapters)} chapters, but {num_chapters} were requested. Adjusting.")
            # If parsing yields fewer chapters, generate generic ones to meet the count
            while len(chapters) < num_chapters:
                chapters.append(f"Chapter {len(chapters) + 1}")
            # If parsing yields more chapters, truncate to requested number
            chapters = chapters[:num_chapters]


        # --- Step 2: Generate Content for Each Chapter Iteratively ---
        # Allocate words per chapter, aiming for roughly equal distribution,
        # but stopping if total limit is reached.
        words_per_chapter_target = (word_limit - current_word_count) // len(chapters) if len(chapters) > 0 else (word_limit - current_word_count)
        
        for i, chapter_title in enumerate(chapters):
            if current_word_count >= word_limit:
                break
            
            # Determine max words for this chapter, considering overall limit
            max_words_for_this_chapter = min(max_words_for_this_chapter, word_limit - current_word_count) # Ensure it doesn't exceed overall limit
            if max_words_for_this_chapter <= 0:
                break

            chapter_prompt = (
                f"Continue writing the {book_type} book titled '{topic}'. Write content for the chapter titled '{chapter_title}'. "
                f"Aim for approximately {max_words_for_this_chapter} words. Ensure the content is cohesive, engaging, "
                "and suitable for a full book chapter. Maintain the style of a {book_type}. Do not add any introductory or concluding remarks, just the chapter content. "
                "Focus purely on the chapter narrative/information."
            )
            chat_history = [{"role": "user", "parts": [{"text": chapter_prompt}]}]
            payload = {"contents": chat_history}

            chapter_response = requests.post(
                GEMINI_TEXT_API_URL,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            chapter_response.raise_for_status()
            
            # Debugging: Print raw response from Gemini API
            print(f"DEBUG: Gemini Book Chapter Raw Response: {chapter_response.text}")

            chapter_result = chapter_response.json()

            chapter_text = ""
            if chapter_result.get('candidates') and chapter_result['candidates'][0].get('content') and chapter_result['candidates'][0]['content'].get('parts'):
                chapter_text = chapter_result['candidates'][0]['content']['parts'][0]['text']
                book_content.append(f"\n\n## {chapter_title}\n\n{chapter_text}\n")
                current_word_count += len(chapter_text.split())
            else:
                book_content.append(f"\n\n## {chapter_title}\n\n(Failed to generate content for this chapter.)\n")
                print(f"Warning: Failed to generate content for chapter '{chapter_title}'.")
        
        final_book_text = "".join(book_content)
        final_word_count = len(final_book_text.split())

        return jsonify({"bookContent": final_book_text, "wordCount": final_word_count})

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API for book generation: {e}")
        # Debugging: Print response content if available for request errors
        if e.response is not None:
            print(f"DEBUG: Gemini Book Error Response Content: {e.response.text}")
        return jsonify({"error": "Failed to generate book via Gemini API", "details": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error during book generation: {e}")
        return jsonify({"error": "An unexpected error occurred during book generation", "details": str(e)}), 500


@app.route('/generate_cover', methods=['POST'])
def generate_cover():
    """
    Generates a book cover image based on prompt using Pollinations.AI.
    Returns the image URL directly.
    """
    try:
        data = request.json
        prompt_text = data.get('prompt')

        if not prompt_text:
            return jsonify({"error": "Prompt for cover generation is required"}), 400

        # Construct the Pollinations.AI URL
        # We need to URL-encode the prompt text
        from urllib.parse import quote_plus
        encoded_prompt = quote_plus(prompt_text)
        image_url = f"{POLLINATIONS_IMAGE_API_BASE_URL}{encoded_prompt}"

        # Pollinations.AI returns the image directly, so we just return the URL to the frontend.
        # The frontend will then display this URL.
        return jsonify({"imageUrl": image_url})

    except Exception as e:
        print(f"Unexpected error during cover generation with Pollinations.AI: {e}")
        return jsonify({"error": "An unexpected error occurred during cover generation", "details": str(e)}), 500

# No changes needed here for PythonAnywhere deployment.
# The 'app' object is already globally defined and will be imported by WSGI.
# The `if __name__ == '__main__':` block is only for local development.
