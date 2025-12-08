
# applications/config.py# GCS Configuration
# Removed: GOOGLE_APPLICATION_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_PATH")
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
import torch
from enum import Enum
from pydantic import BaseModel, Field as PydanticField, validator
from typing import Optional, List

# Load environment variables
load_dotenv('D:/z-whissle/meta-asr/.env') # Keep this for other env vars like NEXT_PUBLIC_API_URL
# load_dotenv("/home/dchauhan/workspace/meta-asr/.env")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISSLE_AUTH_TOKEN = os.getenv("WHISSLE_AUTH_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
TARGET_SAMPLE_RATE = 16000

# Set TEMP_DOWNLOAD_DIR: use env var if set, else create/use ./temp_gcs_downloads in project root
_env_temp_dir = os.getenv("TEMP_DOWNLOAD_DIR")
if _env_temp_dir:
    TEMP_DOWNLOAD_DIR = _env_temp_dir
else:
    # Use a temp_gcs_downloads directory in the root of the project
    TEMP_DOWNLOAD_DIR = str(Path(__file__).parent.parent / "temp_gcs_downloads")

# Ensure the temporary download directory exists
try:
    Path(TEMP_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create temporary download directory {TEMP_DOWNLOAD_DIR}: {e}")

ENTITY_TYPES = [
    "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE",
    "DATE", "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME",
    "EVENT", "MEETING", "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT",
    "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA", "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE",
    "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD",
    "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER",
    "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER",
    "POLICY_HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE",
    "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE",
    "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER",
    "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD",
    "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME",
    "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME",
    "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN"
]

INTENT_TYPES = [
    "INFORM", "QUESTION", "REQUEST", "COMMAND", "GREETING", "CONFIRMATION", "NEGATION",
    "ACKNOWLEDGEMENT", "INQUIRY", "FAREWELL", "APOLOGY", "THANKS", "COMPLAINT",
    "FEEDBACK", "SUGGESTION", "ASSISTANCE", "NAVIGATION", "TRANSACTION", "SCHEDULING",
    "UNKNOWN_INTENT" # Added for robust handling
]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Temporary download directory for GCS files: {TEMP_DOWNLOAD_DIR}")

# Ensure the temporary download directory exists
try:
    Path(TEMP_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create temporary download directory {TEMP_DOWNLOAD_DIR}: {e}")

# Model choice enum
class ModelChoice(str, Enum):
    gemini = "gemini"
    whissle = "whissle"
    deepgram = "deepgram"

# LLM Annotation Model Choice enum
class LlmAnnotationModelChoice(str, Enum):
    gemini = "gemini"
    openai = "openai"
    ollama = "ollama"

USER_API_KEY_TTL_SECONDS = 60 * 60   # 60 minutes

class UserApiKey(BaseModel):
    provider: str
    key: str

class InitSessionRequest(BaseModel):
    user_id: str
    api_keys: List[UserApiKey]

# Pydantic models for requests and responses
class ProcessRequest(BaseModel):
    user_id: str = PydanticField(..., description="Unique identifier for the user.", example="user_123")
    directory_path: str = PydanticField(..., description="Absolute path to the directory containing audio files.", example="/path/to/audio")
    transcriber_choice: Optional[str] = PydanticField(
        None, description="The transcription model to use (whissle, gemini, deepgram). Takes precedence over model_choice if both are provided.",
        example="whissle"
    )
    model_choice: Optional[ModelChoice] = PydanticField(
        None, description="[DEPRECATED] The transcription model to use. Use transcriber_choice instead. Kept for backward compatibility.",
        example="whissle"
    )
    output_jsonl_path: str = PydanticField(..., description="Absolute path for the output JSONL file.", example="/path/to/output/results.jsonl")
    annotations: Optional[List[str]] = PydanticField(
        None, description="List of annotations to include (age, gender, emotion, entity, intent).",
        example=["age", "gender", "emotion", "entity", "intent"]
    )
    llm_annotation_model: Optional[LlmAnnotationModelChoice] = PydanticField(
        None, description="The LLM model to use for annotation (entity, intent). Currently only 'gemini' is supported. Defaults to 'gemini' if not provided.",
        example="gemini"
    )
    prompt: Optional[str] = PydanticField(  # New field
        None, description="Custom prompt for annotation, used when transcription type is annotated. If not provided and entity/intent annotations are requested, a default prompt will be used.",
        example="Transcribe and annotate the audio with BIO tags and intent."
    )
    segment_length_sec: Optional[float] = PydanticField( # New field for trimming
        None, description="Desired length of audio segments in seconds. If provided, audio will be trimmed. Defaults to 30 seconds if not specified.",
        example=30.0
    )
    segment_overlap_sec: Optional[float] = PydanticField( # New field for trimming overlap
        None, description="Overlap between audio segments in seconds. Defaults to 10 seconds if not provided.",
        example=10.0
    )

class TranscriptionJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None
    duration: Optional[float] = None
    model_used_for_transcription: str
    error: Optional[str] = None

class BioAnnotation(BaseModel):
    tokens: List[str]
    tags: List[str]

class AnnotatedJsonlRecord(BaseModel):
    audio_filepath: str
    text: Optional[str] = None
    original_transcription: Optional[str] = None
    duration: Optional[float] = None
    task_name: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    emotion: Optional[str] = None
    gemini_intent: Optional[str] = None
    ollama_intent: Optional[str] = None
    bio_annotation_gemini: Optional[BioAnnotation] = None
    bio_annotation_ollama: Optional[BioAnnotation] = None
    prompt_used: Optional[str] = None  # New field
    annotation_model_error: Optional[str] = None  # Error message when unsupported LLM annotation model is requested
    error: Optional[str] = None

class GcsProcessRequest(BaseModel):
    user_id: str = PydanticField(..., description="Unique identifier for the user.", example="user_123")
    gcs_path: str = PydanticField(..., description="Full GCS path to the audio file (e.g., gs://bucket_name/path/to/audio.wav).", example="gs://your-bucket/audio.wav")
    transcriber_choice: Optional[str] = PydanticField(
        None, description="The transcription model to use (whissle, gemini, deepgram). Takes precedence over model_choice if both are provided.",
        example="whissle"
    )
    model_choice: Optional[ModelChoice] = PydanticField(
        None, description="[DEPRECATED] The transcription model to use. Use transcriber_choice instead. Kept for backward compatibility.",
        example="whissle"
    )
    output_jsonl_path: str = PydanticField(..., description="Absolute path for the output JSONL file where GCS processing results will be saved.", example="/path/to/output/gcs_results.jsonl") # New field
    annotations: Optional[List[str]] = PydanticField(
        None, description="List of annotations to include (age, gender, emotion, entity, intent).",
        example=["age", "gender", "emotion"]
    )
    llm_annotation_model: Optional[LlmAnnotationModelChoice] = PydanticField(
        None, description="The LLM model to use for annotation (entity, intent). Currently only 'gemini' is supported. Defaults to 'gemini' if not provided.",
        example="gemini"
    )
    prompt: Optional[str] = PydanticField(
        None, description="Custom prompt for annotation, used if entity/intent annotations are requested. If not provided, a default prompt will be used.",
        example="Focus on medical entities."
    )

class SingleFileProcessResponse(BaseModel):
    original_gcs_path: str
    downloaded_local_path: Optional[str] = None
    status_message: str
    duration: Optional[float] = None
    transcription: Optional[str] = None
    age_group: Optional[str] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    bio_annotation_gemini: Optional[BioAnnotation] = None # Reusing existing BioAnnotation model
    gemini_intent: Optional[str] = None
    prompt_used: Optional[str] = None
    error_details: Optional[List[str]] = None # To capture a list of errors if multiple steps fail
    overall_error: Optional[str] = None # A summary error message

class ProcessResponse(BaseModel):
    message: str
    output_file: str
    processed_files: int
    saved_records: int
    errors: int


# Removed strict prompt validator - prompt is now optional with default fallback
# The route handlers will use a default prompt from annotation.py if none is provided
# @validator('prompt')
# def validate_prompt(cls, v, values):
#     annotations = values.get('annotations')
#     if annotations and any(a in ['entity', 'intent'] for a in annotations) and not v:
#         raise ValueError("Prompt is required when entity or intent annotations are selected")
#     return v