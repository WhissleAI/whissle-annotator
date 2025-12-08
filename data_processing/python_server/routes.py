# applications/routes.py
from fastapi import FastAPI, HTTPException, APIRouter, WebSocket # Added WebSocket
from fastapi.responses import FileResponse
from pathlib import Path
import gc
import os
import torch
import asyncio
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from config import (
    ProcessRequest, ProcessResponse,
    TranscriptionJsonlRecord, AnnotatedJsonlRecord,
    logger, device,
    ModelChoice, LlmAnnotationModelChoice,
    TARGET_SAMPLE_RATE,
    BioAnnotation,
    InitSessionRequest,
    UserApiKey,
    GcsProcessRequest, # Added
    SingleFileProcessResponse # Added
)
from models import (
    GEMINI_AVAILABLE, WHISSLE_AVAILABLE, DEEPGRAM_AVAILABLE, # Updated to _AVAILABLE flags
    age_gender_model, age_gender_processor,
    emotion_model, emotion_feature_extractor
    # Removed GEMINI_CONFIGURED, WHISSLE_CONFIGURED, DEEPGRAM_CONFIGURED as they are replaced by session checks
)
from audio_utils import validate_paths, discover_audio_files, load_audio, get_audio_duration, trim_audio # Added trim_audio
from transcription import transcribe_with_whissle_single, transcribe_with_gemini_single, transcribe_with_deepgram_single
from annotation import annotate_text_structured_with_gemini
import json
from session_store import init_user_session, is_user_session_valid, get_user_api_key # Added session_store imports
from gcs_utils import parse_gcs_path, download_gcs_blob # Added
from websocket_utils import manager as websocket_manager # Added WebSocket manager
from gcs_batch_processing import process_gcs_directory
from gcs_single_file_processing import _process_single_downloaded_file



router = APIRouter()

def resolve_transcriber_choice(request: ProcessRequest) -> ModelChoice:
    """
    Resolves the transcription model choice from request.
    Prioritizes transcriber_choice over deprecated model_choice.
    Raises HTTPException if neither is provided or value is invalid.
    """
    # Check transcriber_choice first (new field)
    if request.transcriber_choice:
        try:
            return ModelChoice(request.transcriber_choice.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transcriber_choice: '{request.transcriber_choice}'. Must be one of: whissle, gemini, deepgram"
            )
    
    # Fall back to deprecated model_choice for backward compatibility
    if request.model_choice:
        return request.model_choice
    
    # Neither provided
    raise HTTPException(
        status_code=400,
        detail="Either 'transcriber_choice' or 'model_choice' must be provided."
    )

def resolve_transcriber_choice_gcs(request: GcsProcessRequest) -> ModelChoice:
    """
    Resolves the transcription model choice from GCS request.
    Prioritizes transcriber_choice over deprecated model_choice.
    Raises HTTPException if neither is provided or value is invalid.
    """
    # Check transcriber_choice first (new field)
    if request.transcriber_choice:
        try:
            return ModelChoice(request.transcriber_choice.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transcriber_choice: '{request.transcriber_choice}'. Must be one of: whissle, gemini, deepgram"
            )
    
    # Fall back to deprecated model_choice for backward compatibility
    if request.model_choice:
        return request.model_choice
    
    # Neither provided
    raise HTTPException(
        status_code=400,
        detail="Either 'transcriber_choice' or 'model_choice' must be provided."
    )

@router.post("/init_session/", summary="Initialize or Update User API Key Session", status_code=200)
async def init_session_endpoint(session_request: InitSessionRequest):
    try:
        init_user_session(session_request.user_id, session_request.api_keys)
        return {"message": f"Session initialized/updated for user {session_request.user_id}"}
    except Exception as e:
        logger.error(f"Error initializing session for user {session_request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {str(e)}")

def predict_age_gender(audio_data, sampling_rate) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    if age_gender_model is None or age_gender_processor is None:
        return None, None, "Age/Gender model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, None, "Empty audio data provided for Age/Gender."
    try:
        inputs = age_gender_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = age_gender_model(input_values)
        age_pred = outputs[1].detach().cpu().numpy().flatten()[0]
        gender_logits = outputs[2].detach().cpu().numpy()
        gender_pred_idx = np.argmax(gender_logits, axis=1)[0]
        return float(age_pred), int(gender_pred_idx), None
    except Exception as e:
        logger.error(f"Error during Age/Gender prediction: {e}", exc_info=False)
        return None, None, f"Age/Gender prediction failed: {type(e).__name__}"

def predict_emotion(audio_data, sampling_rate) -> Tuple[Optional[str], Optional[str]]:
    if emotion_model is None or emotion_feature_extractor is None:
        return None, "Emotion model not loaded."
    if audio_data is None or len(audio_data) == 0:
        return None, "Empty audio data provided for Emotion."
    min_length = int(sampling_rate * 0.1)
    if len(audio_data) < min_length:
        return "SHORT_AUDIO", None
    try:
        inputs = emotion_feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        emotion_label = emotion_model.config.id2label.get(predicted_class_idx, "UNKNOWN_EMOTION")
        return emotion_label, None
    except Exception as e:
        logger.error(f"Error during Emotion prediction: {e}", exc_info=False)
        return None, f"Emotion prediction failed: {type(e).__name__}"

@router.post("/create_transcription_manifest/", response_model=ProcessResponse, summary="Create Transcription-Only Manifest")
async def create_transcription_manifest_endpoint(process_request: ProcessRequest):
    user_id = process_request.user_id
    if not is_user_session_valid(user_id):
        raise HTTPException(status_code=401, detail="User session is invalid or expired. Please re-initialize session.")

    model_choice = resolve_transcriber_choice(process_request)
    provider_name = model_choice.value

    # Check service availability and user key
    service_available = False
    if model_choice == ModelChoice.whissle:
        service_available = WHISSLE_AVAILABLE
    elif model_choice == ModelChoice.gemini:
        service_available = GEMINI_AVAILABLE
    elif model_choice == ModelChoice.deepgram:
        service_available = DEEPGRAM_AVAILABLE

    if not service_available:
        raise HTTPException(status_code=400, detail=f"{provider_name.capitalize()} SDK is not available on the server.")

    if not get_user_api_key(user_id, provider_name):
        raise HTTPException(status_code=400, detail=f"API key for {provider_name.capitalize()} not found for user or session expired.")

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(message=f"No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0

    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error: Optional[str] = None
                transcription_text: Optional[str] = None
                duration: Optional[float] = None
                logger.info(f"--- Processing {audio_file.name} (Transcription Only for user {user_id}) ---")
                processed_files_count += 1
                try:
                    duration = get_audio_duration(audio_file)
                    if model_choice == ModelChoice.whissle:
                        transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file, user_id)
                    elif model_choice == ModelChoice.gemini:
                        transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file, user_id)
                    elif model_choice == ModelChoice.deepgram:
                        transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file, user_id)
                    else:
                        transcription_error = "Invalid model choice."

                    if transcription_error:
                        file_error = f"Transcription failed: {transcription_error}"
                    elif transcription_text is None: # Explicitly check for None after successful call
                        file_error = "Transcription returned None without an error."

                except Exception as e:
                    logger.error(f"Unexpected error processing {audio_file.name} for user {user_id}: {e}", exc_info=True)
                    file_error = f"Unexpected error: {type(e).__name__}: {e}"
                
                record = TranscriptionJsonlRecord(
                    audio_filepath=str(audio_file.resolve()), 
                    text=transcription_text, 
                    duration=duration, 
                    model_used_for_transcription=model_choice.value, 
                    error=file_error
                )
                try:
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write record for {audio_file.name}: {write_e}", exc_info=True)
                    file_error = (file_error + "; " if file_error else "") + f"JSONL write error: {write_e}"
                
                if file_error:
                    error_count += 1
                else:
                    saved_records_count += 1
                gc.collect()
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")
    
    msg = f"Processed {processed_files_count}/{len(audio_files)}. Saved: {saved_records_count}. Errors: {error_count}."
    return ProcessResponse(message=msg, output_file=str(output_jsonl_path), processed_files=processed_files_count, saved_records=saved_records_count, errors=error_count)


@router.post("/trim_audio_and_transcribe/", response_model=ProcessResponse, summary="Trim Audio Files and Create Transcription Manifesto")
async def trim_audio_and_transcribe_endpoint(process_request: ProcessRequest):
    user_id = process_request.user_id
    if not is_user_session_valid(user_id):
        raise HTTPException(status_code=401, detail="User session is invalid or expired. Please re-initialize session.")

    model_choice = resolve_transcriber_choice(process_request)
    provider_name = model_choice.value
    segment_length_sec = process_request.segment_length_sec or 30.0  # Default to 30 seconds
    segment_overlap_sec = process_request.segment_overlap_sec or 10.0  # Default to 10 seconds

    if segment_length_sec <= 0:
        raise HTTPException(status_code=400, detail="Invalid segment length provided. Must be greater than 0.")
    if segment_overlap_sec < 0:
        raise HTTPException(status_code=400, detail="Invalid segment overlap provided. Must be >= 0.")
    if segment_overlap_sec >= segment_length_sec:
        raise HTTPException(status_code=400, detail="Segment overlap must be less than segment length.")
    
    segment_length_ms = int(segment_length_sec * 1000)
    overlap_ms = int(segment_overlap_sec * 1000)

    # Check service availability and user key
    service_available = False
    if model_choice == ModelChoice.whissle:
        service_available = WHISSLE_AVAILABLE
    elif model_choice == ModelChoice.gemini:
        service_available = GEMINI_AVAILABLE
    elif model_choice == ModelChoice.deepgram:
        service_available = DEEPGRAM_AVAILABLE

    if not service_available:
        raise HTTPException(status_code=400, detail=f"{provider_name.capitalize()} SDK is not available on the server.")

    if not get_user_api_key(user_id, provider_name):
        raise HTTPException(status_code=400, detail=f"API key for {provider_name.capitalize()} not found for user or session expired.")

    dir_path, _ = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    original_audio_files = discover_audio_files(dir_path)

    if not original_audio_files:
        output_jsonl_path = dir_path / "transcriptions.jsonl"
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(message="No audio files found in the directory.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0
    trimmed_audio_base_dir = dir_path / "trimmed_segments"
    trimmed_audio_base_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in original_audio_files:
        try:
            # Create a unique subdirectory for each original file's segments and transcriptions
            file_specific_dir = trimmed_audio_base_dir / audio_file.stem
            file_specific_dir.mkdir(parents=True, exist_ok=True)
            output_jsonl_path = file_specific_dir / "transcriptions.jsonl"

            # Trim audio
            trimmed_segments = await asyncio.to_thread(trim_audio, audio_file, segment_length_ms, file_specific_dir, overlap_ms)
            if not trimmed_segments:
                logger.warning(f"No segments created for {audio_file.name}")
                error_count += 1
                continue

            # Transcribe segments
            with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
                for audio_segment_path in trimmed_segments:
                    file_error: Optional[str] = None
                    transcription_text: Optional[str] = None
                    duration: Optional[float] = None
                    logger.info(f"--- Processing segment {audio_segment_path.name} (Transcription for user {user_id}) ---")
                    processed_files_count += 1
                    try:
                        duration = get_audio_duration(audio_segment_path)
                        if model_choice == ModelChoice.whissle:
                            transcription_text, transcription_error = await transcribe_with_whissle_single(audio_segment_path, user_id)
                        elif model_choice == ModelChoice.gemini:
                            transcription_text, transcription_error = await transcribe_with_gemini_single(audio_segment_path, user_id)
                        elif model_choice == ModelChoice.deepgram:
                            transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_segment_path, user_id)
                        else:
                            transcription_error = "Invalid model choice."

                        if transcription_error:
                            file_error = f"Transcription failed: {transcription_error}"
                        elif transcription_text is None:
                            file_error = "Transcription returned None without an error."

                    except Exception as e:
                        logger.error(f"Unexpected error processing segment {audio_segment_path.name}: {e}", exc_info=True)
                        file_error = f"Unexpected error: {type(e).__name__}: {e}"

                    record = TranscriptionJsonlRecord(
                        audio_filepath=str(audio_segment_path.resolve()),
                        text=transcription_text,
                        duration=duration,
                        model_used_for_transcription=model_choice.value,
                        error=file_error
                    )
                    try:
                        outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                    except Exception as write_e:
                        logger.error(f"Failed to write record for {audio_segment_path.name}: {write_e}", exc_info=True)
                        file_error = (file_error + "; " if file_error else "") + f"JSONL write error: {write_e}"

                    if file_error:
                        error_count += 1
                    else:
                        saved_records_count += 1
                    gc.collect()

        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}", exc_info=True)
            error_count += 1

    msg = f"Processed {processed_files_count} audio segments across {len(original_audio_files)} files. Saved: {saved_records_count}. Errors: {error_count}."
    return ProcessResponse(message=msg, output_file=str(trimmed_audio_base_dir), processed_files=processed_files_count, saved_records=saved_records_count, errors=error_count)





# trim transcribe_annotate_endpoint
@router.post("/trim_transcribe_annotate/", response_model=ProcessResponse, summary="Trim Audio, Transcribe, and Optionally Annotate")
async def trim_transcribe_annotate_endpoint(process_request: ProcessRequest):
    """
    Trims audio files into segments, transcribes them using the selected model, and optionally annotates
    transcriptions with BIO tags and intent using Gemini. Stores results in a JSONL file.
    
    Args:
        process_request: Contains user_id, directory_path, optional output_jsonl_path, model_choice,
                         segment_length_sec, optional annotations list, and optional custom prompt.
    
    Returns:
        ProcessResponse with processing statistics and output file path.
    
    Raises:
        HTTPException: For invalid session, unavailable services, missing API keys, or file I/O errors.
    """
    user_id = process_request.user_id
    if not is_user_session_valid(user_id):
        raise HTTPException(status_code=401, detail="User session is invalid or expired. Please re-initialize session.")

    model_choice = resolve_transcriber_choice(process_request)
    provider_name = model_choice.value
    segment_length_sec = process_request.segment_length_sec or 30.0  # Default to 30 seconds
    segment_overlap_sec = process_request.segment_overlap_sec or 10.0  # Default to 10 seconds

    if segment_length_sec <= 0:
        raise HTTPException(status_code=400, detail="Invalid segment length provided. Must be greater than 0.")
    if segment_overlap_sec < 0:
        raise HTTPException(status_code=400, detail="Invalid segment overlap provided. Must be >= 0.")
    if segment_overlap_sec >= segment_length_sec:
        raise HTTPException(status_code=400, detail="Segment overlap must be less than segment length.")
    
    segment_length_ms = int(segment_length_sec * 1000)
    overlap_ms = int(segment_overlap_sec * 1000)

    # Check transcription service availability and user key
    transcription_service_available = {
        ModelChoice.whissle: WHISSLE_AVAILABLE,
        ModelChoice.gemini: GEMINI_AVAILABLE,
        ModelChoice.deepgram: DEEPGRAM_AVAILABLE
    }.get(model_choice, False)
    
    if not transcription_service_available:
        raise HTTPException(status_code=400, detail=f"{provider_name.capitalize()} SDK is not available on the server.")
    
    if not get_user_api_key(user_id, provider_name):
        raise HTTPException(status_code=400, detail=f"API key for {provider_name.capitalize()} not found or session expired.")

    # Resolve LLM annotation model (default to gemini if not provided)
    llm_annotation_model = process_request.llm_annotation_model or LlmAnnotationModelChoice.gemini
    
    # Check Gemini availability for annotation if needed
    requires_annotation = process_request.annotations and any(a in ["entity", "intent"] for a in process_request.annotations)
    requires_gemini_for_annotation = requires_annotation and llm_annotation_model == LlmAnnotationModelChoice.gemini
    if requires_gemini_for_annotation:
        if not GEMINI_AVAILABLE:
            raise HTTPException(status_code=400, detail="Gemini SDK for annotation is not available on the server.")
        if not get_user_api_key(user_id, "gemini"):
            raise HTTPException(status_code=400, detail="Gemini API key for annotation not found or session expired.")

    needs_age_gender = process_request.annotations and any(a in ["age", "gender"] for a in process_request.annotations)
    needs_emotion = process_request.annotations and "emotion" in process_request.annotations

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    original_audio_files = discover_audio_files(dir_path)

    if not original_audio_files:
        output_jsonl_path = dir_path / "transcriptions.jsonl"
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(
            message="No audio files found in the directory.",
            output_file=str(output_jsonl_path),
            processed_files=0,
            saved_records=0,
            errors=0
        )

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0
    trimmed_audio_base_dir = dir_path / "trimmed_segments"
    trimmed_audio_base_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in original_audio_files:
        try:
            # Create a unique subdirectory for each original file's segments and transcriptions
            file_specific_dir = trimmed_audio_base_dir / audio_file.stem
            file_specific_dir.mkdir(parents=True, exist_ok=True)
            output_jsonl_path = file_specific_dir / "transcriptions.jsonl"

            # Trim audio
            trimmed_segments = await asyncio.to_thread(trim_audio, audio_file, segment_length_ms, file_specific_dir, overlap_ms)
            if not trimmed_segments:
                logger.warning(f"No segments created for {audio_file.name}")
                error_count += 1
                continue

            # Process segments (transcription and optional annotation)
            with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
                for audio_segment_path in trimmed_segments:
                    file_error_details: List[str] = []
                    record_data: Dict[str, Any] = {
                        "audio_filepath": str(audio_segment_path.resolve()),
                        "task_name": "NER" if requires_gemini_for_annotation else "TRANSCRIPTION"
                    }
                    logger.info(f"--- Processing segment {audio_segment_path.name} for user {user_id} ---")
                    processed_files_count += 1

                    # Get audio duration
                    duration = get_audio_duration(audio_segment_path)
                    record_data["duration"] = duration

                    # Load audio for age/gender/emotion if needed
                    audio_data: Optional[np.ndarray] = None
                    sample_rate: Optional[int] = None
                    if needs_age_gender or needs_emotion:
                        audio_data, sample_rate, load_err = load_audio(audio_segment_path)
                        if load_err:
                            file_error_details.append(load_err)
                        elif audio_data is None or (sample_rate != TARGET_SAMPLE_RATE if sample_rate is not None else True):
                            file_error_details.append("Audio load/SR mismatch for A/G/E.")

                    # Transcribe segment
                    transcription_text: Optional[str] = None
                    transcription_error: Optional[str] = None
                    try:
                        if model_choice == ModelChoice.whissle:
                            transcription_text, transcription_error = await transcribe_with_whissle_single(audio_segment_path, user_id)
                        elif model_choice == ModelChoice.gemini:
                            transcription_text, transcription_error = await transcribe_with_gemini_single(audio_segment_path, user_id)
                        elif model_choice == ModelChoice.deepgram:
                            transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_segment_path, user_id)
                        else:
                            transcription_error = "Invalid model choice for transcription."
                        
                        if transcription_error:
                            file_error_details.append(f"Transcription: {transcription_error}")
                        elif transcription_text is None:
                            file_error_details.append("Transcription returned None without an error.")
                    
                    except Exception as e:
                        logger.error(f"Unexpected error transcribing {audio_segment_path.name}: {e}", exc_info=True)
                        file_error_details.append(f"Transcription error: {type(e).__name__}: {e}")

                    record_data["original_transcription"] = transcription_text
                    record_data["text"] = transcription_text  # May be overwritten by annotation

                    # Age/Gender and Emotion processing
                    if (needs_age_gender or needs_emotion) and audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
                        tasks = []
                        if needs_age_gender and 'age_gender_model' in globals():
                            tasks.append(asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE))
                        if needs_emotion and 'emotion_model' in globals():
                            tasks.append(asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE))
                        
                        if tasks:
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                            for result_item in results:
                                if isinstance(result_item, Exception):
                                    logger.error(f"Error in A/G/E sub-task: {result_item}", exc_info=False)
                                    file_error_details.append(f"A_G_E_SubtaskError: {type(result_item).__name__}")
                                    continue
                                
                                if isinstance(result_item, tuple) and len(result_item) == 3:  # Age/Gender
                                    age_pred, gender_idx, age_gender_err = result_item
                                    if age_gender_err:
                                        file_error_details.append(f"A/G_WARN: {age_gender_err}")
                                    else:
                                        if process_request.annotations and "age" in process_request.annotations and age_pred is not None:
                                            actual_age = round(age_pred, 1)
                                            age_brackets = [(18, "0-17"), (25, "18-24"), (35, "25-34"), (45, "35-44"), (55, "45-54"), (65, "55-64"), (float('inf'), "65+")]
                                            age_group = "Unknown"
                                            for threshold, bracket in age_brackets:
                                                if actual_age < threshold:
                                                    age_group = bracket
                                                    break
                                            record_data["age_group"] = age_group
                                        if process_request.annotations and "gender" in process_request.annotations and gender_idx is not None:
                                            gender_str = "Unknown"
                                            if gender_idx == 1:
                                                gender_str = "Male"
                                            elif gender_idx == 0:
                                                gender_str = "Female"
                                            record_data["gender"] = gender_str
                                elif isinstance(result_item, tuple) and len(result_item) == 2:  # Emotion
                                    emotion_label, emotion_err = result_item
                                    if emotion_err:
                                        file_error_details.append(f"EMO_WARN: {emotion_err}")
                                    elif process_request.annotations and "emotion" in process_request.annotations and emotion_label is not None:
                                        record_data["emotion"] = emotion_label.replace("_", " ").title() if emotion_label != "SHORT_AUDIO" else "Short Audio"

                    # LLM Annotation for BIO tags and intent
                    if requires_annotation and transcription_text and transcription_text.strip():
                        if llm_annotation_model == LlmAnnotationModelChoice.gemini:
                            # Use custom prompt if provided, otherwise annotation function will use default prompt
                            prompt_type_for_gemini = process_request.prompt if process_request.prompt else None
                            tokens, tags, intent, gemini_anno_err = await annotate_text_structured_with_gemini(
                                transcription_text,
                                prompt_type_for_gemini, 
                                user_id,
                                
                            )

                            if gemini_anno_err:
                                file_error_details.append(f"GEMINI_ANNOTATION_FAIL: {gemini_anno_err}")
                                if process_request.annotations and "intent" in process_request.annotations:
                                    record_data["gemini_intent"] = "ANNOTATION_FAILED"
                            else:
                                if process_request.annotations and "entity" in process_request.annotations and tokens and tags:
                                    record_data["bio_annotation_gemini"] = BioAnnotation(tokens=tokens, tags=tags)
                                if process_request.annotations and "intent" in process_request.annotations and intent:
                                    record_data["gemini_intent"] = intent
                                record_data["prompt_used"] = prompt_type_for_gemini[:100] if prompt_type_for_gemini else "default_generated_prompt_behavior"
                        else:
                            # Unsupported LLM annotation model
                            error_msg = f"Annotation model '{llm_annotation_model.value}' is not supported; please use 'gemini' for now."
                            file_error_details.append(error_msg)
                            record_data["annotation_model_error"] = error_msg
                            if process_request.annotations and "intent" in process_request.annotations:
                                record_data["gemini_intent"] = "MODEL_NOT_SUPPORTED"

                    elif requires_annotation and (not transcription_text or not transcription_text.strip()):
                        if llm_annotation_model == LlmAnnotationModelChoice.gemini:
                            if process_request.annotations and "intent" in process_request.annotations:
                                record_data["gemini_intent"] = "NO_SPEECH_FOR_ANNOTATION"
                        else:
                            error_msg = f"Annotation model '{llm_annotation_model.value}' is not supported; please use 'gemini' for now."
                            file_error_details.append(error_msg)
                            record_data["annotation_model_error"] = error_msg

                    # Save record
                    final_error_msg = "; ".join(file_error_details) if file_error_details else None
                    record_data["error"] = final_error_msg
                    try:
                        record = AnnotatedJsonlRecord(**record_data)
                        outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                        if not final_error_msg:
                            saved_records_count += 1
                        else:
                            error_count += 1
                    except Exception as write_e:
                        logger.error(f"Failed to write record for {audio_segment_path.name}: {write_e}", exc_info=True)
                        file_error_details.append(f"JSONL write error: {write_e}")
                        record_data["error"] = "; ".join(file_error_details) if file_error_details else write_e
                        error_count += 1
                        outfile.write(json.dumps(record_data, ensure_ascii=False) + "\n")
                    
                    del audio_data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}", exc_info=True)
            error_count += 1

    final_message = (
        f"Processed {processed_files_count} audio segments across {len(original_audio_files)} files. "
        f"Saved {saved_records_count} records successfully. Errors: {error_count}."
    )
    return ProcessResponse(
        message=final_message,
        output_file=str(trimmed_audio_base_dir),
        processed_files=processed_files_count,
        saved_records=saved_records_count,
        errors=error_count
    )


# only for annotated manifest
@router.post("/create_annotated_manifest/", response_model=ProcessResponse, summary="Create Annotated Manifest")
async def create_annotated_manifest_endpoint(process_request: ProcessRequest):
    user_id = process_request.user_id
    if not is_user_session_valid(user_id):
        raise HTTPException(status_code=401, detail="User session is invalid or expired. Please re-initialize session.")

    model_choice = resolve_transcriber_choice(process_request)
    transcription_provider_name = model_choice.value

    # Check transcription model availability and user key
    transcription_service_available = False
    if model_choice == ModelChoice.whissle:
        transcription_service_available = WHISSLE_AVAILABLE
    elif model_choice == ModelChoice.gemini:
        transcription_service_available = GEMINI_AVAILABLE
    elif model_choice == ModelChoice.deepgram:
        transcription_service_available = DEEPGRAM_AVAILABLE

    if not transcription_service_available:
        raise HTTPException(status_code=400, detail=f"{transcription_provider_name.capitalize()} SDK for transcription is not available on the server.")
    # if not get_user_api_key(user_id, transcription_provider_name):
    #     raise HTTPException(status_code=400, detail=f"API key for {transcription_provider_name.capitalize()} (transcription) not found for user or session expired.")

    # Resolve LLM annotation model (default to gemini if not provided)
    llm_annotation_model = process_request.llm_annotation_model or LlmAnnotationModelChoice.gemini
    
    # Check Gemini availability for annotation if needed
    requires_annotation = process_request.annotations and any(a in ["entity", "intent"] for a in process_request.annotations)
    requires_gemini_for_annotation = requires_annotation and llm_annotation_model == LlmAnnotationModelChoice.gemini
    if requires_gemini_for_annotation:
        if not GEMINI_AVAILABLE:
            raise HTTPException(status_code=400, detail="Gemini SDK for annotation is not available on the server.")
        # if not get_user_api_key(user_id, "gemini"):
        #     raise HTTPException(status_code=400, detail="Gemini API key for annotation not found for user or session expired.")

    logger.info(f"User {user_id} - Received annotated request with prompt: {process_request.prompt[:100] if process_request.prompt else 'None'}")

    needs_age_gender = process_request.annotations and any(a in ["age", "gender"] for a in process_request.annotations)
    needs_emotion = process_request.annotations and "emotion" in process_request.annotations

    dir_path, output_jsonl_path = validate_paths(process_request.directory_path, process_request.output_jsonl_path)
    audio_files = discover_audio_files(dir_path)

    if not audio_files:
        try:
            with open(output_jsonl_path, "w", encoding="utf-8") as _: pass
        except IOError as e: 
            raise HTTPException(status_code=500, detail=f"Failed to create empty output file: {e}")
        return ProcessResponse(message=f"No audio files. Empty manifest created.", output_file=str(output_jsonl_path), processed_files=0, saved_records=0, errors=0)

    processed_files_count = 0
    saved_records_count = 0
    error_count = 0

    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for audio_file in audio_files:
                file_error_details: List[str] = []
                record_data: Dict[str, Any] = {"audio_filepath": str(audio_file.resolve()), "task_name": "NER"}
                logger.info(f"--- User {user_id} - Processing {audio_file.name} (Selective Annotation) ---")
                processed_files_count += 1
                record_data["duration"] = get_audio_duration(audio_file)
                audio_data: Optional[np.ndarray] = None # Ensure type hint for audio_data
                sample_rate: Optional[int] = None

                if needs_age_gender or needs_emotion:
                    audio_data, sample_rate, load_err = load_audio(audio_file)
                    if load_err:
                        file_error_details.append(load_err)
                    elif audio_data is None or (sample_rate != TARGET_SAMPLE_RATE if sample_rate is not None else True):
                        file_error_details.append("Audio load/SR mismatch for A/G/E.")
                
                transcription_text: Optional[str] = None
                transcription_error: Optional[str] = None

                if model_choice == ModelChoice.whissle:
                    transcription_text, transcription_error = await transcribe_with_whissle_single(audio_file, user_id)
                elif model_choice == ModelChoice.gemini:
                    transcription_text, transcription_error = await transcribe_with_gemini_single(audio_file, user_id)
                elif model_choice == ModelChoice.deepgram:
                    transcription_text, transcription_error = await transcribe_with_deepgram_single(audio_file, user_id)
                else:
                    transcription_error = "Invalid model choice for transcription."

                if transcription_error:
                    file_error_details.append(f"Transcription: {transcription_error}")
                    # transcription_text remains None
                elif transcription_text is None: # Explicitly check for None after successful call
                    file_error_details.append("Transcription returned None without an error.")
                
                record_data["original_transcription"] = transcription_text
                record_data["text"] = transcription_text # This might be overwritten by annotation if successful

                # Age/Gender and Emotion processing (remains largely the same, ensure audio_data and sample_rate are valid)
                if (needs_age_gender or needs_emotion) and audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
                    tasks = []
                    if needs_age_gender:
                        if age_gender_model is None: file_error_details.append("A/G_WARN: Age/Gender model not loaded.")
                        else: tasks.append(asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE))
                    if needs_emotion:
                        if emotion_model is None: file_error_details.append("EMO_WARN: Emotion model not loaded.")
                        else: tasks.append(asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE))
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result_item in results: # Renamed result to result_item to avoid conflict
                            if isinstance(result_item, Exception):
                                logger.error(f"Error in A/G/E sub-task: {result_item}", exc_info=False)
                                file_error_details.append(f"A_G_E_SubtaskError: {type(result_item).__name__}")
                                continue

                            if isinstance(result_item, tuple) and len(result_item) == 3: # Age/Gender result
                                age_pred, gender_idx, age_gender_err = result_item
                                if age_gender_err: file_error_details.append(f"A/G_WARN: {age_gender_err}")
                                else:
                                    if process_request.annotations and "age" in process_request.annotations and age_pred is not None:
                                        try:
                                            actual_age = round(age_pred, 1)
                                            age_brackets = [(18, "0-17"), (25, "18-24"), (35, "25-34"), (45, "35-44"), (55, "45-54"), (65, "55-64"), (float('inf'), "65+")]
                                            age_group = "Unknown"
                                            for threshold, bracket in age_brackets:
                                                if actual_age < threshold:
                                                    age_group = bracket
                                                    break
                                            record_data["age_group"] = age_group
                                        except Exception as age_e:
                                            logger.error(f"Error formatting age_group: {age_e}")
                                            record_data["age_group"] = "Error"
                                    if process_request.annotations and "gender" in process_request.annotations and gender_idx is not None:
                                        gender_str = "Unknown"
                                        if gender_idx == 1: gender_str = "Male"
                                        elif gender_idx == 0: gender_str = "Female"
                                        record_data["gender"] = gender_str
                            elif isinstance(result_item, tuple) and len(result_item) == 2: # Emotion result
                                emotion_label, emotion_err = result_item
                                if emotion_err: file_error_details.append(f"EMO_WARN: {emotion_err}")
                                elif process_request.annotations and "emotion" in process_request.annotations and emotion_label is not None:
                                    record_data["emotion"] = emotion_label.replace("_", " ").title() if emotion_label != "SHORT_AUDIO" else "Short Audio"
                
                # LLM Annotation
                if requires_annotation and transcription_text and transcription_text.strip() != "":
                    if llm_annotation_model == LlmAnnotationModelChoice.gemini:
                        if process_request.annotations and ("entity" in process_request.annotations or "intent" in process_request.annotations):
                            # Use user's custom prompt if provided, otherwise annotation function will use default prompt
                            prompt_type_for_gemini = process_request.prompt if process_request.prompt else None
                        
                        tokens, tags, intent, gemini_anno_err = await annotate_text_structured_with_gemini(
                            transcription_text, 
                            custom_prompt=prompt_type_for_gemini, 
                            user_id=user_id
                        )
                        if gemini_anno_err:
                            file_error_details.append(f"GEMINI_ANNOTATION_FAIL: {gemini_anno_err}")
                            if process_request.annotations and "intent" in process_request.annotations: record_data["gemini_intent"] = "ANNOTATION_FAILED"
                        else:
                            if process_request.annotations and "entity" in process_request.annotations and tokens and tags:
                                 record_data["bio_annotation_gemini"] = BioAnnotation(tokens=tokens, tags=tags)
                            if process_request.annotations and "intent" in process_request.annotations and intent:
                                 record_data["gemini_intent"] = intent
                        record_data["prompt_used"] = prompt_type_for_gemini[:100] if prompt_type_for_gemini else "default_generated_prompt_behavior" # Clarify what prompt was used
                    else:
                        # Unsupported LLM annotation model
                        error_msg = f"Annotation model '{llm_annotation_model.value}' is not supported; please use 'gemini' for now."
                        file_error_details.append(error_msg)
                        record_data["annotation_model_error"] = error_msg
                        if process_request.annotations and "intent" in process_request.annotations:
                            record_data["gemini_intent"] = "MODEL_NOT_SUPPORTED"
                
                elif requires_annotation: # Case where transcription failed or was empty
                    if llm_annotation_model == LlmAnnotationModelChoice.gemini:
                        if not transcription_text or transcription_text.strip() == "":
                            if process_request.annotations and "intent" in process_request.annotations: record_data["gemini_intent"] = "NO_SPEECH_FOR_ANNOTATION"
                    else:
                        error_msg = f"Annotation model '{llm_annotation_model.value}' is not supported; please use 'gemini' for now."
                        file_error_details.append(error_msg)
                        record_data["annotation_model_error"] = error_msg
                    # else: # This case should be covered by transcription_error already
                        # if process_request.annotations and "intent" in process_request.annotations: record_data["gemini_intent"] = "TRANSCRIPTION_FAILED_FOR_ANNOTATION"

                final_error_msg = "; ".join(file_error_details) if file_error_details else None
                record_data["error"] = final_error_msg
                
                # ... (rest of record saving and error counting logic remains similar) ...
                current_errors_before_write = error_count
                try:
                    record = AnnotatedJsonlRecord(**record_data)
                    outfile.write(record.model_dump_json(exclude_none=True) + "\n")
                except Exception as write_e:
                    logger.error(f"Failed to write annotated record for {audio_file.name}: {write_e}", exc_info=True)
                    final_error_msg = (final_error_msg + "; " if final_error_msg else "") + f"JSONL write error: {write_e}"
                    record_data["error"] = final_error_msg # Update record_data if write fails
                    if error_count == current_errors_before_write: # Ensure error is counted if write fails
                        error_count += 1
                
                if not final_error_msg: # Successfully processed and written
                    saved_records_count += 1
                elif error_count == current_errors_before_write and final_error_msg: # Error occurred before write, or write itself failed
                     error_count +=1 # Ensure error is counted if it happened before write and wasn't already

                del audio_data # Cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ... (final message and return ProcessResponse remains similar) ...
        truly_successful_saves = saved_records_count
        final_message = (
            f"Processed {processed_files_count}/{len(audio_files)} files for selective annotation. "
            f"{truly_successful_saves} records successfully saved (no internal errors). "
            f"{error_count} files encountered errors or warnings (check 'error' field in JSONL)."
        )
        return ProcessResponse(
            message=final_message,
            output_file=str(output_jsonl_path),
            processed_files=processed_files_count,
            saved_records=truly_successful_saves,
            errors=error_count
        )
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write output file: {e}")








@router.post("/process_gcs_file/", response_model=SingleFileProcessResponse, summary="Download, Chunk, Transcribe, and Optionally Annotate GCS File")
async def process_gcs_file_endpoint(request: GcsProcessRequest):
    """
    Downloads an audio file from GCS, chunks it if >30s, transcribes, and optionally annotates segments.
    Saves results to a JSONL file and returns processing details.

    Args:
        request: Contains user_id, gcs_path, output_jsonl_path, model_choice, annotations, and optional prompt.

    Returns:
        SingleFileProcessResponse with processing results and status.
    """
    user_id = request.user_id
    # The output path
    """
    Downloads an audio file from GCS, chunks it if >30s, transcribes, and optionally annotates segments.
    Saves results to a JSONL file and returns processing details.
    
    Args:
        request: Contains user_id, gcs_path, output_jsonl_path, model_choice, annotations, and optional prompt.
    
    Returns:
        SingleFileProcessResponse with processing results and status.
    """
    user_id = request.user_id
    model_choice = resolve_transcriber_choice_gcs(request)
    # Handle both absolute and relative paths for output_jsonl_path
    output_jsonl_path = Path(request.output_jsonl_path)
    if not output_jsonl_path.is_absolute():
        # If relative, make it relative to a base output directory
        base_output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", str(Path(__file__).parent.parent / "outputs")))
        output_jsonl_path = base_output_dir / output_jsonl_path
        logger.info(f"User {user_id} - Relative path provided, using base output dir: {base_output_dir}")
    logger.info(f"User {user_id} - Received request to process GCS file: {request.gcs_path}. Output: {output_jsonl_path}")
   
    await websocket_manager.send_personal_message({"status": "request_received", "detail": f"Received request for {request.gcs_path}"}, user_id)

    # if not is_user_session_valid(user_id):
    #     logger.warning(f"User {user_id} - Invalid or expired session.")
    #     await websocket_manager.send_personal_message({"status": "error", "detail": "User session is invalid or expired."}, user_id)
    #     return SingleFileProcessResponse(
    #         original_gcs_path=request.gcs_path,
    #         status_message="User session is invalid or expired. Please re-initialize session.",
    #         overall_error="AuthenticationError"
    #     )

    bucket_name, blob_name = parse_gcs_path(request.gcs_path)
    if not bucket_name or not blob_name:
        logger.error(f"User {user_id} - Invalid GCS path: {request.gcs_path}")
        await websocket_manager.send_personal_message({"status": "error", "detail": "Invalid GCS path format."}, user_id)
        return SingleFileProcessResponse(
            original_gcs_path=request.gcs_path,
            status_message="Invalid GCS path format.",
            overall_error="InvalidInput"
        )

    # Ensure output directory exists
# Note: The actual subdirectory will be created in _process_single_downloaded_file
    try:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"User {user_id} - Ensured base output directory exists: {output_jsonl_path.parent}")
    except Exception as e:
        logger.error(f"User {user_id} - Failed to create base output directory {output_jsonl_path.parent}: {e}")
        await websocket_manager.send_personal_message({"status": "error", "detail": f"Failed to create base output directory: {str(e)}"}, user_id)
        return SingleFileProcessResponse(
            original_gcs_path=request.gcs_path,
            status_message=f"Failed to create base output directory: {output_jsonl_path.parent}",
            overall_error="OutputDirectoryError"
        )

    await websocket_manager.send_personal_message({"status": "download_started", "detail": f"Downloading gs://{bucket_name}/{blob_name}..."}, user_id)
    local_audio_path: Optional[Path] = None
    try:
        local_audio_path = await asyncio.to_thread(download_gcs_blob, bucket_name, blob_name)
        if not local_audio_path:
            logger.error(f"User {user_id} - Failed to download gs://{bucket_name}/{blob_name}")
            err_msg = f"Failed to download file from GCS path: gs://{bucket_name}/{blob_name}"
            await websocket_manager.send_personal_message({"status": "download_failed", "detail": err_msg}, user_id)
            return SingleFileProcessResponse(
                original_gcs_path=request.gcs_path,
                status_message=err_msg,
                overall_error="DownloadError"
            )
        logger.info(f"User {user_id} - Successfully downloaded to {local_audio_path}")
        await websocket_manager.send_personal_message({"status": "download_complete", "detail": f"Successfully downloaded to {local_audio_path}"}, user_id)

        processing_results = await _process_single_downloaded_file(
            local_audio_path,
            user_id,
            model_choice,
            request.annotations,
            request.llm_annotation_model,
            request.prompt,
            output_jsonl_path,
            request.gcs_path,
            request.segment_length_sec,
            request.segment_overlap_sec
        )

        status_msg = f"File processed. Results saved to {output_jsonl_path}"
        if processing_results["overall_error_summary"]:
            status_msg = f"File processed with errors: {processing_results['overall_error_summary']}"
        elif not processing_results["transcription"]:
            status_msg = "File processed, but no transcriptions were generated."

        
        def get_first_item(val):
            if isinstance(val, list) and val:
                return val[0]
            return "" if val == [] or val is None else val
        def get_first_dict(val):
            if isinstance(val, list) and val:
                return val[0]
            return None
        return SingleFileProcessResponse(
            original_gcs_path=request.gcs_path,
            downloaded_local_path=str(local_audio_path),
            status_message=status_msg,
            duration=processing_results.get("duration"),
            transcription=get_first_item(processing_results.get("transcription")),
            age_group=get_first_item(processing_results.get("age_group")),
            gender=get_first_item(processing_results.get("gender")),
            emotion=get_first_item(processing_results.get("emotion")),
            bio_annotation_gemini=get_first_dict(processing_results.get("bio_annotation_gemini")),
            gemini_intent=get_first_item(processing_results.get("gemini_intent")),
            # prompt_used=processing_results.get("prompt_used"),
            error_details=processing_results.get("error_details"),
            overall_error=processing_results.get("overall_error_summary")
        )

    except Exception as e:
        logger.error(f"User {user_id} - Unexpected error processing GCS file {request.gcs_path}: {e}", exc_info=True)
        err_msg = f"Unexpected server error: {type(e).__name__}"
        await websocket_manager.send_personal_message({"status": "error", "detail": err_msg}, user_id)
        return SingleFileProcessResponse(
            original_gcs_path=request.gcs_path,
            downloaded_local_path=str(local_audio_path) if local_audio_path else None,
            status_message=err_msg,
            overall_error="ServerError"
        )
    finally:
        if local_audio_path and local_audio_path.exists():
            try:
                local_audio_path.unlink()
                logger.info(f"User {user_id} - Cleaned up temporary file: {local_audio_path}")
                await websocket_manager.send_personal_message({"status": "cleanup", "detail": f"Cleaned up temporary file: {local_audio_path}"}, user_id)
            except Exception as e:
                logger.error(f"User {user_id} - Failed to clean up {local_audio_path}: {e}")
                await websocket_manager.send_personal_message({"status": "cleanup_failed", "detail": f"Failed to clean up temporary file: {str(e)}"}, user_id)




# handle directory
@router.post("/process_gcs_directory/", summary="Process all .wav files in a GCS directory (bucket folder)")
async def process_gcs_directory_endpoint(request: GcsProcessRequest):
    """
    Processes all .wav files in a GCS directory: downloads, transcribes/annotates, and saves results.
    Args:
        request: Contains user_id, gcs_path (directory), output_jsonl_path, model_choice, annotations, prompt, path_type.
    Returns:
        Dict with processing summary.
    """
    user_id = request.user_id
    model_choice = resolve_transcriber_choice_gcs(request)
    # Handle both absolute and relative paths for output_jsonl_path
    output_jsonl_path = Path(request.output_jsonl_path)
    if not output_jsonl_path.is_absolute():
        # If relative, make it relative to a base output directory
        base_output_dir = Path(os.getenv("DEFAULT_OUTPUT_DIR", str(Path(__file__).parent.parent / "outputs")))
        output_jsonl_path = base_output_dir / output_jsonl_path
        logger.info(f"User {user_id} - Relative path provided, using base output dir: {base_output_dir}")
    logger.info(f"User {user_id} - Received request to process GCS directory: {request.gcs_path}. Output: {output_jsonl_path}")
    await websocket_manager.send_personal_message({"status": "request_received", "detail": f"Received directory request for {request.gcs_path}"}, user_id)

    # if not is_user_session_valid(user_id):
    #     logger.warning(f"User {user_id} - Invalid or expired session.")
    #     await websocket_manager.send_personal_message({"status": "error", "detail": "User session is invalid or expired."}, user_id)
    #     return {"error": "User session is invalid or expired. Please re-initialize session."}

    try:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"User {user_id} - Failed to create base output directory {output_jsonl_path.parent}: {e}")
        await websocket_manager.send_personal_message({"status": "error", "detail": f"Failed to create base output directory: {str(e)}"}, user_id)
        return {"error": f"Failed to create base output directory: {output_jsonl_path.parent}"}

    # Call the batch processing function
    results = await process_gcs_directory(
        user_id=user_id,
        gcs_dir_path=request.gcs_path,
        model_choice=model_choice,
        requested_annotations=request.annotations,
        llm_annotation_model=request.llm_annotation_model,
        custom_prompt=request.prompt,
        output_jsonl_path=output_jsonl_path,
        path_type=getattr(request, "path_type", None),
        segment_length_sec=request.segment_length_sec,
        segment_overlap_sec=request.segment_overlap_sec
    )
    return results


@router.websocket("/ws/gcs_status/{user_id}")
async def gcs_status_websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket_manager.connect(websocket, user_id)
    logger.info(f"User {user_id} WebSocket connected for GCS status updates.")
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logger.info(f"User {user_id} WebSocket disconnected or error: {type(e).__name__} - {e}")
    finally:
        websocket_manager.disconnect(user_id)
        logger.info(f"User {user_id} WebSocket connection closed.")




@router.get("/status", summary="API Status")
async def get_status():
    return {
        "message": "Welcome to the Audio Processing API (User Session Based)",
        "docs_url": "/docs", "html_interface": "/",
        "endpoints": {
            "init_session": "/init_session/",
            "transcription_only": "/create_transcription_manifest/",
            "full_annotation": "/create_annotated_manifest/"
        },
        "gemini_sdk_available": GEMINI_AVAILABLE,
        "whissle_sdk_available": WHISSLE_AVAILABLE,
        "deepgram_sdk_available": DEEPGRAM_AVAILABLE,
        "age_gender_model_loaded": age_gender_model is not None,
        "emotion_model_loaded": emotion_model is not None,
        "device": str(device)
    }