# app.py
import sys
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from murf import Murf
import io
import requests
import json
import base64
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- API Key Configuration ---
MURF_API_KEY = os.environ.get("MURF_API_KEY", "YOUR_MURF_AI_API_KEY_FOR_LOCAL_TESTING")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_FOR_LOCAL_TESTING")

POLLINATIONS_IMAGE_API_BASE_URL = "https://image.pollinations.ai/prompt/"

# --- Initialize API Clients/URLs ---
try:
    client = Murf(api_key=MURF_API_KEY)
    print(f"DEBUG: Murf client initialized successfully. API Key (first 5 chars): {MURF_API_KEY[:5]}...", file=sys.stderr)
except Exception as e:
    print(f"ERROR: Failed to initialize Murf client: {e}", file=sys.stderr)
    client = None

GEMINI_TEXT_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

print(f"DEBUG: App starting. MURF_API_KEY (first 5 chars): {MURF_API_KEY[:5]}...", file=sys.stderr)
print(f"DEBUG: GEMINI_API_KEY (first 5 chars): {GEMINI_API_KEY[:5]}...", file=sys.stderr)


@app.route('/')
def serve_index():
    """Serves the index.html (or home.html) file."""
    return send_file('index.html') # Assuming index.html is now the main landing page

# Serve other HTML files for each tab
@app.route('/tts.html')
def serve_tts():
    return send_file('tts.html')

@app.route('/humanizer.html')
def serve_humanizer():
    return send_file('humanizer.html')

@app.route('/book_writer.html')
def serve_book_writer():
    return send_file('book_writer.html')

@app.route('/cover_generator.html')
def serve_cover_generator():
    return send_file('cover_generator.html')

@app.route('/feedback.html')
def serve_feedback_page(): # Renamed to avoid conflict with endpoint
    return send_file('feedback.html')


@app.route('/voices', methods=['GET'])
def get_voices():
    """
    Retrieves a list of available Murf AI voices.
    IMPORTANT: Please refer to Murf AI's official Voice Library
    to find the exact 'voice_id's that are available and
    active for your account, and update this list accordingly.
    Visit: https://murf.ai/api/docs/voices-styles/voice-library to view the voice library in our API docs.
    """
    print("DEBUG: Received request for /voices endpoint.", file=sys.stderr)
    if client is None:
        print("ERROR: Murf client is not initialized. Cannot fetch voices.", file=sys.stderr)
        return jsonify({"error": "Murf AI client not initialized. Check API key."}), 500

    try:
        voices_list = [
            {"name": "Aarav (Indian English, Male, Conversational)", "id": "en-IN-aarav", "gender": "Male", "locale": "en-IN", "voice_type": "Conversational"},
            {"name": "Arohi (Indian English, Female, Conversational)", "id": "en-IN-arohi", "gender": "Female", "locale": "en-IN", "voice_type": "Conversational"},
            {"name": "Rohan (Indian English, Male, Conversational)", "id": "en-IN-rohan", "gender": "Male", "locale": "en-IN", "voice_type": "Conversational"},
            {"name": "Alia (Indian English, Female, Promo)", "id": "en-IN-alia", "gender": "Female", "locale": "en-IN", "voice_type": "Promo"},
            # Removed Surya as requested
            {"name": "Priya (Indian English, Female, Conversational)", "id": "en-IN-priya", "gender": "Female", "locale": "en-IN", "voice_type": "Conversational"},
            {"name": "Shivani (Indian English, Female, Conversational)", "id": "en-IN-shivani", "gender": "Female", "locale": "en-IN", "voice_type": "Conversational"},
            {"name": "Isha (Indian English, Female, Conversational)", "id": "en-IN-isha", "gender": "Female", "locale": "en-IN", "voice_type": "Conversational"},
            {"name": "Eashwar (Indian English, Male, Conversational)", "id": "en-IN-eashwar", "gender": "Male", "locale": "en-IN", "voice_type": "Conversational"},
            {"name": "Natalie (US English, Female, Neural)", "id": "en-US-natalie", "gender": "Female", "locale": "en-US", "voice_type": "Neural"},
            {"name": "Liam (Australian English, Male, Neural)", "id": "en-AU-liam", "gender": "Male", "locale": "en-AU", "voice_type": "Neural"},
        ]
        
        voices_list.sort(key=lambda v: (v['locale'], v['name']))
        print("DEBUG: Successfully prepared voices list (Surya removed).", file=sys.stderr)
        return jsonify(voices_list)
    except Exception as e:
        print(f"ERROR: Exception in get_voices: {e}", file=sys.stderr)
        return jsonify({"error": "Failed to retrieve voices", "details": str(e)}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """
    Synthesizes text into speech using Murf AI.
    """
    print("DEBUG: Received request for /synthesize endpoint.", file=sys.stderr)
    if client is None:
        print("ERROR: Murf client is not initialized. Cannot synthesize speech.", file=sys.stderr)
        return jsonify({"error": "Murf AI client not initialized. Check API key."}), 500

    try:
        data = request.json
        text = data.get('text')
        voice_id = data.get('voiceId')
        pitch_val = float(data.get('pitch', 0.0))
        speed_val = float(data.get('speed', 1.0))
        action = data.get('action')

        if not text or not voice_id:
            print("ERROR: Missing text or voice ID for synthesis.", file=sys.stderr)
            return jsonify({"error": "Missing text or voice ID"}), 400

        murf_pitch = int(pitch_val)
        if speed_val < 1.0:
            murf_rate = int(((speed_val - 0.25) / 0.75) * 50 - 50)
        elif speed_val > 1.0:
            murf_rate = int(((speed_val - 1.0) / 3.0) * 50)
        else:
            murf_rate = 0
        murf_rate = max(-50, min(50, murf_rate))

        print(f"DEBUG: Calling Murf AI generate with text length {len(text)}, voice_id={voice_id}, pitch={murf_pitch}, rate={murf_rate}", file=sys.stderr)
        response = client.text_to_speech.generate(
            text=text,
            voice_id=voice_id,
            format="MP3",
            rate=murf_rate,
            pitch=murf_pitch
        )
        
        audio_url = response.audio_file

        if not audio_url:
            print("ERROR: Murf AI did not return an audio file URL.", file=sys.stderr)
            raise Exception("Murf AI did not return an audio file URL.")

        print(f"DEBUG: Murf AI returned audio URL: {audio_url}", file=sys.stderr)
        audio_response = requests.get(audio_url, stream=True)
        audio_response.raise_for_status()

        audio_buffer = io.BytesIO(audio_response.content)

        if action == 'download':
            print("DEBUG: Sending audio for download.", file=sys.stderr)
            return send_file(
                audio_buffer,
                mimetype='audio/mpeg',
                as_attachment=True,
                download_name='converted_audio.mp3'
            )
        else:
            print("DEBUG: Sending audio for playback.", file=sys.stderr)
            return audio_buffer.getvalue(), 200, {'Content-Type': 'audio/mpeg'}

    except Exception as e:
        print(f"ERROR: Error during Murf AI speech synthesis: {e}", file=sys.stderr)
        return jsonify({"error": "Speech synthesis failed with Murf AI", "details": str(e)}), 500

@app.route('/humanize_text', methods=['POST'])
def humanize_text():
    """
    Takes text from the user and humanizes it using the Gemini API.
    """
    print("DEBUG: Received request for /humanize_text endpoint.", file=sys.stderr)
    try:
        data = request.json
        text_to_humanize = data.get('text')

        if not text_to_humanize:
            print("ERROR: No text provided for humanization.", file=sys.stderr)
            return jsonify({"error": "No text provided for humanization"}), 400

        prompt = f"Humanize the following text, making it sound 100% natural and written by a human. Ensure it is grammatically correct, flows well, and uses appropriate tone and vocabulary. Do not add any introductory or concluding remarks, just the humanized text:\n\n{text_to_humanize}"
        
        chat_history = []
        chat_history.append({ "role": "user", "parts": [{ "text": prompt }] })
        
        payload = { "contents": chat_history }

        gemini_response = requests.post(
            GEMINI_TEXT_API_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        gemini_response.raise_for_status()
        
        print(f"DEBUG: Gemini Humanize Raw Response Status: {gemini_response.status_code}", file=sys.stderr)

        result = gemini_response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            humanized_text = result['candidates'][0]['content']['parts'][0]['text']
            print("DEBUG: Successfully humanized text.", file=sys.stderr)
            return jsonify({"humanizedText": humanized_text})
        else:
            print(f"ERROR: Gemini API response for humanize format unexpected or missing content: {result}", file=sys.stderr)
            raise Exception("Gemini API response format unexpected or missing content.")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error calling Gemini API for humanization: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"DEBUG: Gemini Humanize Error Response Content: {e.response.text}", file=sys.stderr)
        return jsonify({"error": "Failed to humanize text via Gemini API", "details": str(e)}), 500
    except Exception as e:
        print(f"ERROR: Unexpected error during text humanization: {e}", file=sys.stderr)
        return jsonify({"error": "An unexpected error occurred during humanization", "details": str(e)}), 500

@app.route('/generate_book', methods=['POST'])
def generate_book():
    """
    Generates a book with chapters based on topic/title, type, and word/chapter limits.
    """
    print("DEBUG: Received request for /generate_book endpoint.", file=sys.stderr)
    try:
        data = request.json
        topic = data.get('topic')
        word_limit = int(data.get('word_limit', 5000))
        num_chapters = int(data.get('num_chapters', 5))
        book_type = data.get('book_type', 'novel')

        if not topic:
            print("ERROR: Book topic or title is required.", file=sys.stderr)
            return jsonify({"error": "Book topic or title is required"}), 400
        
        if word_limit > 50000:
            word_limit = 50000
            print("DEBUG: Word limit adjusted to max 50,000 words.", file=sys.stderr)
        
        if num_chapters <= 0:
            print("ERROR: Number of chapters must be at least 1.", file=sys.stderr)
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
        
        print("DEBUG: Requesting book outline from Gemini API.", file=sys.stderr)
        outline_response = requests.post(
            GEMINI_TEXT_API_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        outline_response.raise_for_status()
        
        print(f"DEBUG: Gemini Book Outline Raw Response Status: {outline_response.status_code}", file=sys.stderr)

        outline_result = outline_response.json()
        outline_text = ""
        if outline_result.get('candidates') and outline_result['candidates'][0].get('content') and outline_result['candidates'][0]['content'].get('parts'):
            outline_text = outline_result['candidates'][0]['content']['parts'][0]['text']
            book_content.append(f"# {topic}\n\n## Outline\n{outline_text}\n\n")
            current_word_count += len(outline_text.split())
            print("DEBUG: Successfully generated book outline.", file=sys.stderr)
        else:
            print(f"ERROR: Gemini API response for outline format unexpected or missing content: {outline_result}", file=sys.stderr)
            raise Exception("Gemini API response for outline format unexpected or missing content.")

        # --- Parse Chapters from Outline ---
        chapters = []
        lines = outline_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(tuple(f"{i}." for i in range(1, num_chapters + 1))):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    chapter_title = parts[0].strip()
                    chapters.append(chapter_title)
        
        if len(chapters) != num_chapters:
            print(f"WARNING: Parsed {len(chapters)} chapters, but {num_chapters} were requested. Adjusting.", file=sys.stderr)
            while len(chapters) < num_chapters:
                chapters.append(f"Chapter {len(chapters) + 1}")
            chapters = chapters[:num_chapters]
        print(f"DEBUG: Prepared {len(chapters)} chapters for generation.", file=sys.stderr)

        # --- Step 2: Generate Content for Each Chapter Iteratively ---
        words_per_chapter_target = (word_limit - current_word_count) // len(chapters) if len(chapters) > 0 else (word_limit - current_word_count)
        
        for i, chapter_title in enumerate(chapters):
            if current_word_count >= word_limit:
                print("DEBUG: Word limit reached, stopping chapter generation.", file=sys.stderr)
                break
            
            max_words_for_this_chapter = min(words_per_chapter_target, word_limit - current_word_count)
            if max_words_for_this_chapter <= 0:
                print("DEBUG: No words left for current chapter, stopping.", file=sys.stderr)
                break

            chapter_prompt = (
                f"Continue writing the {book_type} book titled '{topic}'. Write content for the chapter titled '{chapter_title}'. "
                f"Aim for approximately {max_words_for_this_chapter} words. Ensure the content is cohesive, engaging, "
                "and suitable for a full book chapter. Maintain the style of a {book_type}. Do not add any introductory or concluding remarks, just the chapter content. "
            )
            chat_history = [{"role": "user", "parts": [{"text": chapter_prompt}]}]
            payload = {"contents": chat_history}

            print(f"DEBUG: Requesting content for chapter '{chapter_title}' from Gemini API.", file=sys.stderr)
            chapter_response = requests.post(
                GEMINI_TEXT_API_URL,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            chapter_response.raise_for_status()
            
            print(f"DEBUG: Gemini Book Chapter Raw Response Status: {chapter_response.status_code}", file=sys.stderr)

            chapter_result = chapter_response.json()

            chapter_text = ""
            if chapter_result.get('candidates') and chapter_result['candidates'][0].get('content') and chapter_result['candidates'][0]['content'].get('parts'):
                chapter_text = chapter_result['candidates'][0]['content']['parts'][0]['text']
                book_content.append(f"\n\n## {chapter_title}\n\n{chapter_text}\n")
                current_word_count += len(chapter_text.split())
                print(f"DEBUG: Generated content for chapter '{chapter_title}'. Current word count: {current_word_count}", file=sys.stderr)
            else:
                book_content.append(f"\n\n## {chapter_title}\n\n(Failed to generate content for this chapter.)\n")
                print(f"WARNING: Failed to generate content for chapter '{chapter_title}'. Response: {chapter_result}", file=sys.stderr)
        
        final_book_text = "".join(book_content)
        final_word_count = len(final_book_text.split())
        print(f"DEBUG: Book generation complete. Final word count: {final_word_count}", file=sys.stderr)
        return jsonify({"bookContent": final_book_text, "wordCount": final_word_count})

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error calling Gemini API for book generation: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"DEBUG: Gemini Book Error Response Content: {e.response.text}", file=sys.stderr)
        return jsonify({"error": "Failed to generate book via Gemini API", "details": str(e)}), 500
    except Exception as e:
        print(f"ERROR: Unexpected error during book generation: {e}", file=sys.stderr)
        return jsonify({"error": "An unexpected error occurred during book generation", "details": str(e)}), 500


@app.route('/generate_cover', methods=['POST'])
def generate_cover():
    """
    Generates a book cover image based on prompt using Pollinations.AI.
    Returns the image URL directly.
    """
    print("DEBUG: Received request for /generate_cover endpoint.", file=sys.stderr)
    try:
        data = request.json
        prompt_text = data.get('prompt')

        if not prompt_text:
            print("ERROR: Prompt for cover generation is required.", file=sys.stderr)
            return jsonify({"error": "Prompt for cover generation is required"}), 400

        from urllib.parse import quote_plus
        encoded_prompt = quote_plus(prompt_text)
        image_url = f"{POLLINATIONS_IMAGE_API_BASE_URL}{encoded_prompt}"
        
        print(f"DEBUG: Pollinations.AI image URL generated: {image_url}", file=sys.stderr)
        return jsonify({"imageUrl": image_url})

    except Exception as e:
        print(f"ERROR: Unexpected error during cover generation with Pollinations.AI: {e}", file=sys.stderr)
        return jsonify({"error": "An unexpected error occurred during cover generation", "details": str(e)}), 500

@app.route('/send_feedback', methods=['POST'])
def send_feedback():
    """
    Receives feedback from the frontend and acknowledges its receipt.
    (Email integration removed as per user request).
    """
    print("DEBUG: Received request for /send_feedback endpoint.", file=sys.stderr)
    try:
        data = request.json
        name = data.get('name', 'Anonymous')
        email = data.get('email', 'No Email Provided')
        subject = data.get('subject', 'General Feedback')
        message_body = data.get('message', 'No message body provided.')

        if not message_body:
            return jsonify({"error": "Message body is required for feedback."}), 400

        # Log the feedback for debugging/record-keeping on the server side
        print(f"INFO: Received Feedback -- Name: {name}, Email: {email}, Subject: {subject}, Message: {message_body}", file=sys.stderr)
        
        return jsonify({"message": "Thank you for your feedback! It has been received."}), 200

    except Exception as e:
        print(f"ERROR: Failed to process feedback: {e}", file=sys.stderr)
        return jsonify({"error": "Failed to receive feedback. Please try again later.", "details": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"DEBUG: Running Flask app on port {port}", file=sys.stderr)
    app.run(host='0.0.0.0', port=port)
