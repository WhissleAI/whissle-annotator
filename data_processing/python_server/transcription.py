


# applications/transcription.py
from pathlib import Path
import asyncio
import os
from typing import Tuple, Optional
import requests
from config import logger, GOOGLE_API_KEY, DEEPGRAM_API_KEY, WHISSLE_STT_AUTH_TOKEN, WHISSLE_STT_API_URL # Removed WHISSLE_AUTH_TOKEN
from models import GEMINI_AVAILABLE, WHISSLE_AVAILABLE, DEEPGRAM_AVAILABLE # Use *_AVAILABLE flags
# Removed: from models import DEEPGRAM_CLIENT
from session_store import get_user_api_key
import google.generativeai as genai
from deepgram import PrerecordedOptions, DeepgramClient as DeepgramSDKClient # Ensure DeepgramSDKClient is used

def get_mime_type(audio_file_path: Path) -> str:
    ext = audio_file_path.suffix.lower()
    mime_map = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4" }
    return mime_map.get(ext, "application/octet-stream")

async def transcribe_with_whissle_single(audio_path: Path, user_id: str, model_name="en-US-0.6b") -> Tuple[Optional[str], Optional[str]]:
    if not WHISSLE_AVAILABLE:
        return None, "Whissle SDK is not available."
    
    whissle_auth_token = get_user_api_key(user_id, "whissle")
    if not whissle_auth_token:
        return None, "Whissle API key not found or session expired for user."

    try:
        from whissle import WhissleClient # Keep local import if preferred
        whissle_client = WhissleClient(auth_token=whissle_auth_token)
        response = await whissle_client.speech_to_text(str(audio_path), model_name=model_name)
        if isinstance(response, dict):
            text = response.get('text')
            if text is not None: return text.strip(), None
            else:
                error_detail = response.get('error') or response.get('message', 'Unknown Whissle API error structure')
                return None, f"Whissle API error: {error_detail}"
        elif hasattr(response, 'transcript') and isinstance(response.transcript, str):
            return response.transcript.strip(), None
        else: return None, f"Unexpected Whissle response format: {type(response)}"
    except Exception as e:
        return None, f"Whissle SDK error: {type(e).__name__}: {e}"


async def transcribe_with_whissle_stt_single(audio_path: Path, user_id: str) -> Tuple[Optional[str], Optional[str]]:
    if not WHISSLE_AVAILABLE:
        return None, "Whissle SDK is not available."

    if not WHISSLE_STT_AUTH_TOKEN:
        return None, "Whissle STT auth token is not configured."

    if not WHISSLE_STT_API_URL:
        return None, "Whissle STT endpoint is not configured."

    def _perform_request():
        with open(audio_path, "rb") as audio_file_obj:
            files = {
                "audio": (audio_path.name, audio_file_obj, get_mime_type(audio_path))
            }
            return requests.post(
                WHISSLE_STT_API_URL,
                params={"auth_token": WHISSLE_STT_AUTH_TOKEN},
                files=files,
                timeout=180
            )

    try:
        response = await asyncio.to_thread(_perform_request)
    except Exception as exc:
        logger.error(f"Whissle STT request failed for {audio_path.name}: {exc}", exc_info=True)
        return None, f"Whissle STT request failed: {type(exc).__name__}: {exc}"

    if response.status_code != 200:
        details = response.text
        try:
            details = response.json()
        except ValueError:
            pass
        logger.error(f"Whissle STT returned {response.status_code}: {details}")
        return None, f"Whissle STT error {response.status_code}: {details}"

    try:
        payload = response.json()
    except ValueError as exc:
        logger.error(f"Failed to parse Whissle STT response for {audio_path.name}: {exc}", exc_info=True)
        return None, "Whissle STT response JSON parsing failed."

    transcript = payload.get("transcript")
    if transcript:
        return transcript.strip(), None
    logger.error(f"Whissle STT response missing transcript: {payload}")
    return None, "Whissle STT response did not include a transcript."

async def transcribe_with_gemini_single(audio_path: Path, user_id: str) -> Tuple[Optional[str], Optional[str]]:
    if not GEMINI_AVAILABLE:
        return None, "Gemini (google.generativeai)/api key is not available."

    # gemini_api_key = get_user_api_key(user_id, "gemini")
    gemini_api_key = GOOGLE_API_KEY
    if not gemini_api_key:
        return None, "Gemini API key not found in environment variables."  # FIXED: Return 2 elements, not 4

    # Critical: genai.configure is a global setting.
    # This can cause issues in concurrent environments if not handled carefully.
    # For true per-request isolation, the Gemini SDK would need to support key-per-client or key-per-request.
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        logger.error(f"Failed to configure Gemini with user API key: {e}")
        return None, "Failed to configure Gemini API with user key."

    model_name = "models/gemini-2.0-flash"
    uploaded_file = None # Ensure uploaded_file is defined for the finally block
    # ... rest of the function remains largely the same, ensure genai calls use the configured key
    logger.info(f"Starting Gemini transcription for {audio_path.name} with model {model_name}...")
    
    # Add file verification
    if not audio_path.exists():
        return None, f"Audio file does not exist: {audio_path}"
    
    file_size = audio_path.stat().st_size
    logger.info(f"Audio file size: {file_size} bytes")
    
    if file_size == 0:
        return None, f"Audio file is empty: {audio_path}"
    
    try:
        model = genai.GenerativeModel(model_name)
        mime_type = get_mime_type(audio_path)
        logger.info(f"Detected MIME type: {mime_type}")
        
        logger.info(f"Uploading file to Gemini: {audio_path}")
        uploaded_file = await asyncio.to_thread(genai.upload_file, path=str(audio_path), mime_type=mime_type)
        logger.info(f"File uploaded successfully. File name: {uploaded_file.name}, Initial state: {uploaded_file.state.name}")
        
        while uploaded_file.state.name == "PROCESSING":
            logger.info("File still processing, waiting...")
            await asyncio.sleep(2) # Consider making sleep duration configurable or dynamic
            uploaded_file = await asyncio.to_thread(genai.get_file, name=uploaded_file.name)
            logger.info(f"File state updated: {uploaded_file.state.name}")
            
        if uploaded_file.state.name != "ACTIVE":
            error_msg = f"Gemini file processing failed for {audio_path.name}. State: {uploaded_file.state.name}"
            # No need to await asyncio.to_thread for genai.delete_file if it's synchronous
            # However, if it's I/O bound and there's a sync version, direct call is fine.
            # Assuming genai.delete_file can be awaited if it's an async operation or wrapped if sync.
            try: 
                # If genai.delete_file is sync, wrap it: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
                # If it's already async: await genai.delete_file(name=uploaded_file.name)
                # For now, assuming it's okay to call directly if it's a quick cleanup.
                # Let's assume it's okay to call directly for cleanup, or it's an async function.
                # If it's a sync blocking call, it should be wrapped.
                # For safety, wrapping:
                await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete failed Gemini resource {uploaded_file.name}: {del_e}")
            return None, error_msg
        
        prompt = """Please transcribe the audio file. Listen carefully to the speech and write down exactly what is being said. 
        
If you can hear speech, provide the complete transcription of all spoken words.
If there is no speech or if the audio is silent, respond with: NO_SPEECH_DETECTED
If you cannot understand the speech clearly, provide your best attempt at transcription.

Transcription:"""
        logger.info(f"Making Gemini API call with prompt: '{prompt[:40]}'")
        
        # The model.generate_content call will use the globally configured API key
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        response = await asyncio.to_thread(
            model.generate_content, 
            [prompt, uploaded_file], 
            request_options={'timeout': 300},
            safety_settings=safety_settings
        )
        logger.info(f"Received response from Gemini API")
        logger.info(f"Response type: {type(response)}")
        
        try: 
            await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            logger.info(f"Successfully deleted uploaded file: {uploaded_file.name}")
        except Exception as del_e: 
            logger.warning(f"Could not delete Gemini resource {uploaded_file.name} after transcription: {del_e}")

        if response.candidates:
            try:
                # Add detailed logging for debugging
                logger.info(f"Gemini response has {len(response.candidates)} candidates")
                # logger.info(f"Response object attributes: {dir(response)}")
                # logger.info(f"First candidate attributes: {dir(response.candidates[0])}")
                
                transcription = None
                if hasattr(response, 'text') and response.text is not None: 
                    transcription = response.text.strip()
                    logger.info(f"Got transcription from response.text: '{transcription}'")
                elif response.candidates[0].content.parts: 
                    transcription = "".join(part.text for part in response.candidates[0].content.parts).strip()
                    logger.info(f"Got transcription from content.parts: '{transcription}'")
                    logger.info(f"Number of parts: {len(response.candidates[0].content.parts)}")
                    for i, part in enumerate(response.candidates[0].content.parts):
                        logger.info(f"Part {i}: '{part.text}'")
                else: 
                    logger.error("Gemini response candidate found, but no text content.")
                    logger.info(f"Candidate content: {response.candidates[0].content}")
                    return None, "Gemini response candidate found, but no text content."
                
                logger.info(f"Final transcription result: '{transcription}' (length: {len(transcription) if transcription else 0})")
                return transcription if transcription else "", None # Return empty string if transcription is empty
            except (AttributeError, IndexError, ValueError, TypeError) as resp_e: # More specific exception handling
                logger.error(f"Error parsing Gemini transcription response for {audio_path.name}: {resp_e}", exc_info=True)
                return None, f"Error parsing Gemini transcription response: {resp_e}"
        else:
            error_message = f"No candidates from Gemini transcription for {audio_path.name}."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                # Check if block_reason_message exists, otherwise convert feedback to string
                feedback = getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback))
                error_message += f" Feedback: {feedback}"
            return None, error_message
    except Exception as e:
        logger.error(f"Gemini API/SDK error during transcription for {audio_path.name}: {e}", exc_info=True)
        if uploaded_file and hasattr(uploaded_file, 'name'): # Check if uploaded_file exists and has a name
            try: await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
            except Exception as del_e: logger.warning(f"Could not delete temp Gemini file {uploaded_file.name} after error: {del_e}")
        return None, f"Gemini API/SDK error: {type(e).__name__}: {e}"
    # finally: # Optional: Reset global genai configuration if necessary, though this is tricky
        # genai.configure(api_key=None) # Or to a system key if available. This is not ideal.


async def transcribe_with_deepgram_single(audio_path: Path, user_id: str) -> Tuple[Optional[str], Optional[str]]:
    if not DEEPGRAM_AVAILABLE:
        return None, "Deepgram SDK is not available."

    deepgram_api_key = get_user_api_key(user_id, "deepgram")
    if not deepgram_api_key:
        deepgram_api_key = DEEPGRAM_API_KEY
    if not deepgram_api_key:
        return None, "Deepgram API key not found in session or environment."

    try:
        # Initialize Deepgram client with the user-specific key
        deepgram_client = DeepgramSDKClient(api_key=deepgram_api_key)
        
        with open(audio_path, "rb") as audio_file_obj: # Renamed to avoid conflict
            buffer_data = audio_file_obj.read()
        
        source = {"buffer": buffer_data, "mimetype": get_mime_type(audio_path)}
        options = PrerecordedOptions(
            model="nova-2", smart_format=True, diarize=False, language="en" # Ensure language is set if needed
        )
        
        # The Deepgram SDK's transcribe_file method might be synchronous or asynchronous.
        # If synchronous, it should be wrapped with asyncio.to_thread.
        # Assuming listen.prerecorded.v("1").transcribe_file is a synchronous blocking call.
        response = await asyncio.to_thread(
            deepgram_client.listen.prerecorded.v("1").transcribe_file,
            source,
            options
        )
        
        # Robust transcript extraction
        if response and response.results and response.results.channels and \
           response.results.channels[0].alternatives and \
           response.results.channels[0].alternatives[0].transcript:
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript.strip() if transcript else "", None # Return empty string if transcript is empty but present
        else:
            logger.warning(f"Deepgram transcription for {audio_path.name} returned unexpected structure or empty transcript. Response: {response}")
            return None, "Deepgram returned no transcript or unexpected response structure."

    except Exception as e:
        logger.error(f"Deepgram transcription error for {audio_path.name}: {e}", exc_info=True)
        return None, f"Deepgram SDK/API error: {type(e).__name__}: {e}"