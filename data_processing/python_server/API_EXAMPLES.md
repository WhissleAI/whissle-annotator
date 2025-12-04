# API Usage Examples for Postman and cURL

This document provides practical examples for using the Audio Processing API with Postman or cURL.

## Base URL

```
http://localhost:8000
```

## Table of Contents

1. [Initialize Session (Optional)](#1-initialize-session-optional)
2. [Process Single GCS File](#2-process-single-gcs-file)
3. [Process GCS Directory](#3-process-gcs-directory)
4. [Check API Status](#4-check-api-status)

---

## 1. Initialize Session (Optional)

**Note:** Session initialization is optional. If you don't initialize a session, the server will use environment variables for API keys (if configured).

### Endpoint
```
POST /init_session/
```

### Request Body
```json
{
  "user_id": "user_123",
  "api_keys": [
    {
      "provider": "gemini",
      "key": "your_gemini_api_key_here"
    },
    {
      "provider": "whissle",
      "key": "your_whissle_key_here"
    },
    {
      "provider": "deepgram",
      "key": "your_deepgram_key_here"
    }
  ]
}
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/init_session/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "api_keys": [
      {"provider": "gemini", "key": "your_gemini_api_key_here"},
      {"provider": "whissle", "key": "your_whissle_key_here}
    ]
  }'
```

### Response
```json
{
  "message": "Session initialized/updated for user user_123"
}
```

---

## 2. Process Single GCS File

Downloads a single audio file from Google Cloud Storage, transcribes it, and optionally annotates it.

### Endpoint
```
POST /process_gcs_file/
```

### Request Body Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | Unique identifier for the user |
| `gcs_path` | string | Yes | Full GCS path (e.g., `gs://bucket-name/path/to/audio.wav`) |
| `model_choice` | string | Yes | Transcription model: `"gemini"`, `"whissle"`, or `"deepgram"` |
| `output_jsonl_path` | string | Yes | Output path (absolute or relative). If relative, will be relative to `DEFAULT_OUTPUT_DIR` or `data_processing/outputs` |
| `annotations` | array | No | List of annotations: `["age", "gender", "emotion", "entity", "intent"]` |
| `prompt` | string | No | Custom prompt for annotation. If not provided and entity/intent annotations are requested, a default prompt will be used |

### Example 1: Basic Transcription Only

```json
{
  "user_id": "user_123",
  "gcs_path": "gs://your-bucket/audio/sample.wav",
  "model_choice": "gemini",
  "output_jsonl_path": "/tmp/output/results"
}
```

### cURL Example 1
```bash
curl -X POST "http://localhost:8000/process_gcs_file/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "gcs_path": "gs://your-bucket/audio/sample.wav",
    "model_choice": "gemini",
    "output_jsonl_path": "/tmp/output/results"
  }'
```

### Example 2: Transcription with All Annotations (Using Default Prompt)

```json
{
  "user_id": "user_123",
  "gcs_path": "gs://your-bucket/audio/sample.wav",
  "model_choice": "gemini",
  "output_jsonl_path": "results",
  "annotations": ["age", "gender", "emotion", "entity", "intent"]
}
```

### cURL Example 2
```bash
curl -X POST "http://localhost:8000/process_gcs_file/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "gcs_path": "gs://your-bucket/audio/sample.wav",
    "model_choice": "gemini",
    "output_jsonl_path": "results",
    "annotations": ["age", "gender", "emotion", "entity", "intent"]
  }'
```

### Example 3: Transcription with Custom Prompt

```json
{
  "user_id": "user_123",
  "gcs_path": "gs://your-bucket/audio/soccer_commentary.wav",
  "model_choice": "gemini",
  "output_jsonl_path": "/tmp/soccer_output",
  "annotations": ["entity", "intent"],
  "prompt": "You are an expert linguistic annotator specializing in soccer commentary. Identify player names, team names, match events, and commentary intent."
}
```

### cURL Example 3
```bash
curl -X POST "http://localhost:8000/process_gcs_file/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "gcs_path": "gs://your-bucket/audio/soccer_commentary.wav",
    "model_choice": "gemini",
    "output_jsonl_path": "/tmp/soccer_output",
    "annotations": ["entity", "intent"],
    "prompt": "You are an expert linguistic annotator specializing in soccer commentary. Identify player names, team names, match events, and commentary intent."
  }'
```

### Response Example
```json
{
  "original_gcs_path": "gs://your-bucket/audio/sample.wav",
  "downloaded_local_path": "/path/to/temp/downloaded_file.wav",
  "status_message": "File processed. Results saved to /tmp/output/results",
  "duration": 124.32,
  "transcription": "to get us underway in the shootout...",
  "age_group": "25-34",
  "gender": "Male",
  "emotion": "Excited",
  "bio_annotation_gemini": {
    "tokens": ["to", "get", "us", "underway", ...],
    "tags": ["B-LIVE_COMMENTARY", "I-LIVE_COMMENTARY", ...]
  },
  "gemini_intent": "LIVE_COMMENTARY",
  "error_details": null,
  "overall_error": null
}
```

---

## 3. Process GCS Directory

Processes all audio files in a GCS directory (folder).

### Endpoint
```
POST /process_gcs_directory/
```

### Request Body Parameters

Same as single file, but `gcs_path` should point to a directory (folder) instead of a file.

### Example 1: Process Directory with Default Prompt

```json
{
  "user_id": "user_123",
  "gcs_path": "gs://your-bucket/audio_folder/",
  "model_choice": "gemini",
  "output_jsonl_path": "/tmp/batch_output",
  "annotations": ["age", "gender", "emotion", "entity", "intent"]
}
```

### cURL Example 1
```bash
curl -X POST "http://localhost:8000/process_gcs_directory/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "gcs_path": "gs://your-bucket/audio_folder/",
    "model_choice": "gemini",
    "output_jsonl_path": "/tmp/batch_output",
    "annotations": ["age", "gender", "emotion", "entity", "intent"]
  }'
```

### Example 2: Process Directory with Custom Soccer Prompt

```json
{
  "user_id": "user_123",
  "gcs_path": "gs://stream2action-audio/youtube-videos/soccer_data",
  "model_choice": "gemini",
  "output_jsonl_path": "soccer_results",
  "annotations": ["entity", "intent"],
  "prompt": "You are an expert linguistic annotator specializing in soccer (football) commentary. Identify PLAYER_NAME, TEAM_NAME, GOAL, PENALTY, and other soccer-specific entities. Use intent types like LIVE_COMMENTARY, GOAL_ANNOUNCEMENT, etc."
}
```

### cURL Example 2
```bash
curl -X POST "http://localhost:8000/process_gcs_directory/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "gcs_path": "gs://stream2action-audio/youtube-videos/soccer_data",
    "model_choice": "gemini",
    "output_jsonl_path": "soccer_results",
    "annotations": ["entity", "intent"],
    "prompt": "You are an expert linguistic annotator specializing in soccer (football) commentary. Identify PLAYER_NAME, TEAM_NAME, GOAL, PENALTY, and other soccer-specific entities."
  }'
```

### Response Example
```json
{
  "message": "Processed 5 files. Saved 5 records successfully. Errors: 0.",
  "output_file": "/tmp/batch_output",
  "processed_files": 5,
  "saved_records": 5,
  "errors": 0
}
```

---

## 4. Check API Status

Check which services and models are available.

### Endpoint
```
GET /status
```

### cURL Example
```bash
curl -X GET "http://localhost:8000/status"
```

### Response Example
```json
{
  "message": "Welcome to the Audio Processing API (User Session Based)",
  "docs_url": "/docs",
  "html_interface": "/",
  "endpoints": {
    "init_session": "/init_session/",
    "transcription_only": "/create_transcription_manifest/",
    "full_annotation": "/create_annotated_manifest/"
  },
  "gemini_sdk_available": true,
  "whissle_sdk_available": true,
  "deepgram_sdk_available": true,
  "age_gender_model_loaded": true,
  "emotion_model_loaded": true,
  "device": "cuda"
}
```

---

## Important Notes

### Output Path Handling

- **Absolute paths**: Use full paths like `/tmp/output` or `D:/output` (Windows)
- **Relative paths**: Will be resolved relative to `DEFAULT_OUTPUT_DIR` environment variable, or `data_processing/outputs` by default
- **Example**: If `output_jsonl_path` is `"results"`, it will be saved to `{DEFAULT_OUTPUT_DIR}/results`

### Prompt Handling

- **With prompt**: If you provide a custom `prompt`, it will be used for entity/intent annotation
- **Without prompt**: If `prompt` is omitted but `annotations` includes `"entity"` or `"intent"`, a default prompt will be automatically used
- **No annotation**: If `annotations` doesn't include `"entity"` or `"intent"`, the `prompt` field is ignored

### Annotations

Available annotation types:
- `"age"` - Predicts age group (0-17, 18-24, 25-34, etc.)
- `"gender"` - Predicts gender (Male/Female)
- `"emotion"` - Predicts emotion (Happy, Sad, Excited, etc.)
- `"entity"` - BIO tagging for named entities (requires prompt or uses default)
- `"intent"` - Intent classification (requires prompt or uses default)

### Model Choices

- `"gemini"` - Google Gemini 2.0 Flash
- `"whissle"` - Whissle transcription service
- `"deepgram"` - Deepgram transcription service

### GCS Path Format

- Single file: `gs://bucket-name/path/to/file.wav`
- Directory: `gs://bucket-name/path/to/folder/` (note the trailing slash)

### WebSocket Status Updates

For real-time status updates, connect to:
```
WS /ws/gcs_status/{user_id}
```

---

## Postman Collection Setup

1. Create a new collection in Postman
2. Set base URL variable: `{{base_url}}` = `http://localhost:8000`
3. Add the endpoints above as requests
4. Use the JSON examples as request bodies
5. Save common values (like `user_id`) as collection variables

---

## Troubleshooting

### Error: "User session is invalid or expired"
- Solution: Call `/init_session/` first, or ensure environment variables have API keys configured

### Error: "Invalid GCS path format"
- Solution: Ensure GCS path starts with `gs://` and includes bucket name and path

### Error: "Failed to create base output directory"
- Solution: Check that the output path is valid and the server has write permissions

### Error: "API key for {provider} not found"
- Solution: Initialize session with API keys or set environment variables

---

## Additional Endpoints

For local file processing (not GCS), see:
- `POST /create_transcription_manifest/` - Transcribe local audio files
- `POST /create_annotated_manifest/` - Transcribe and annotate local audio files
- `POST /trim_transcribe_annotate/` - Trim, transcribe, and annotate local audio files

For more details, visit: `http://localhost:8000/docs` (Swagger UI)





