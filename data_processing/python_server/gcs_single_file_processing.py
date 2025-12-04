import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from config import (
    ModelChoice, LlmAnnotationModelChoice, TARGET_SAMPLE_RATE, BioAnnotation, logger, device
)
from audio_utils import get_audio_duration, trim_audio, load_audio
from session_store import get_user_api_key
from websocket_utils import manager as websocket_manager
from annotation import annotate_text_structured_with_gemini
from models import (
    GEMINI_AVAILABLE, # Updated to _AVAILABLE flags
    age_gender_model, age_gender_processor,
    emotion_model, emotion_feature_extractor 
)
from transcription import transcribe_with_whissle_single, transcribe_with_gemini_single, transcribe_with_deepgram_single


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


# *********

async def _process_single_downloaded_file(
    local_audio_path: Path,
    user_id: str,
    model_choice: ModelChoice,
    requested_annotations: Optional[List[str]],
    llm_annotation_model: Optional[LlmAnnotationModelChoice],
    custom_prompt: Optional[str],
    output_jsonl_path: Path,
    original_gcs_path: str
) -> Dict[str, Any]:
    """
    Processes a single downloaded audio file by chunking (if >30s), transcribing, and optionally annotating.
    Sends progress updates via WebSocket and returns aggregated results for all segments.
    
    Args:
        local_audio_path: Path to the downloaded audio file.
        user_id: User identifier for API key and WebSocket communication.
        model_choice: Selected transcription model (whissle, gemini, deepgram).
        requested_annotations: List of requested annotations (e.g., entity, intent, age, gender, emotion).
        custom_prompt: Optional custom prompt for Gemini annotation.
        output_jsonl_path: Path to save JSONL results.
        original_gcs_path: Original GCS path for reference in output.
    
    Returns:
        Dictionary containing aggregated results and errors across all segments.
    """
    results: Dict[str, Any] = {
        "duration": None,
        "transcription": [],
        "age_group": [],
        "gender": [],
        "emotion": [],
        "bio_annotation_gemini": [],
        "gemini_intent": [],
        "prompt_used": None,
        "error_details": [],
        "overall_error_summary": None
    }
    segment_results = []  # Store results for each segment
    await websocket_manager.send_personal_message({"status": "processing_started", "detail": f"Processing file: {local_audio_path.name}"}, user_id)

    # --- Duration ---
    try:
        await websocket_manager.send_personal_message({"status": "calculating_duration", "detail": "Calculating audio duration..."}, user_id)
        duration = await asyncio.to_thread(get_audio_duration, local_audio_path)
        results["duration"] = duration
        await websocket_manager.send_personal_message({"status": "duration_complete", "detail": f"Duration: {duration:.2f}s"}, user_id)
    except Exception as e:
        logger.error(f"User {user_id} - Failed to get duration for {local_audio_path.name}: {e}")
        results["error_details"].append(f"DurationError: {str(e)}")
        await websocket_manager.send_personal_message({"status": "error", "detail": f"Failed to get duration: {str(e)}"}, user_id)
        return results  # Early return if duration fails

    # --- Trim Audio ---
    segment_length_ms = 30 * 1000  # 30 seconds
    
    # Extract the filename from GCS path (last part after '/')
    gcs_filename = Path(original_gcs_path).stem  # Gets filename without extension
    # Create safe directory name from GCS filename
    safe_gcs_filename = "".join(c for c in gcs_filename if c.isalnum() or c in ['-', '_'])
    if not safe_gcs_filename:  # Fallback if filename becomes empty after sanitization
        safe_gcs_filename = "audio_segments"

    # Create subdirectory inside the user-provided output path
    # output_jsonl_path already contains the full user-provided path
    segments_dir = output_jsonl_path / safe_gcs_filename
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Update paths
    trimmed_audio_dir = segments_dir / "segments"  # Audio segments go in 'segments' subfolder
    trimmed_audio_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl_path = segments_dir / "annotations.jsonl"  # JSONL goes in the main subfolder

    await websocket_manager.send_personal_message({
        "status": "trimming_started", 
        "detail": f"Creating segments directory: {segments_dir}"
    }, user_id)
    try:
        trimmed_segments = await asyncio.to_thread(trim_audio, local_audio_path, segment_length_ms, trimmed_audio_dir)
        if not trimmed_segments:
            results["error_details"].append(f"TrimmingError: No segments created for {local_audio_path.name}")
            await websocket_manager.send_personal_message({"status": "trimming_failed", "detail": "No segments created."}, user_id)
            return results
        await websocket_manager.send_personal_message({"status": "trimming_complete", "detail": f"Created {len(trimmed_segments)} segment(s)."}, user_id)
    except Exception as e:
        logger.error(f"User {user_id} - Failed to trim {local_audio_path.name}: {e}")
        results["error_details"].append(f"TrimmingError: {str(e)}")
        await websocket_manager.send_personal_message({"status": "trimming_failed", "detail": f"Failed to trim audio: {str(e)}"}, user_id)
        return results

    # --- Process Each Segment ---
    transcription_provider_name = model_choice.value
    # Resolve LLM annotation model (default to gemini if not provided)
    resolved_llm_model = llm_annotation_model or LlmAnnotationModelChoice.gemini
    requires_annotation = requested_annotations and any(a in ["entity", "intent"] for a in requested_annotations)
    requires_gemini_for_annotation = requires_annotation and resolved_llm_model == LlmAnnotationModelChoice.gemini
    needs_local_models = requested_annotations and any(a in ["age", "gender", "emotion"] for a in requested_annotations)

    for segment_path in trimmed_segments:
        segment_result: Dict[str, Any] = {
            "segment_path": str(segment_path),
            "duration": None,
            "transcription": None,
            "age_group": None,
            "gender": None,
            "emotion": None,
            "bio_annotation_gemini": None,
            "gemini_intent": None,
            "error_details": []
        }
        transcription_text: Optional[str] = None  # Ensure variable is always defined
        transcription_error: Optional[str] = None
        await websocket_manager.send_personal_message({"status": "segment_processing_started", "detail": f"Processing segment: {segment_path.name}"}, user_id)

        # Segment Duration
        try:
            segment_duration = await asyncio.to_thread(get_audio_duration, segment_path)
            segment_result["duration"] = segment_duration
            await websocket_manager.send_personal_message({"status": "segment_duration_complete", "detail": f"Segment duration: {segment_duration:.2f}s"}, user_id)
        except Exception as e:
            logger.error(f"User {user_id} - Failed to get duration for segment {segment_path.name}: {e}")
            segment_result["error_details"].append(f"SegmentDurationError: {str(e)}")
            await websocket_manager.send_personal_message({"status": "segment_error", "detail": f"Failed to get segment duration: {str(e)}"}, user_id)

        # Transcription
        # if not get_user_api_key(user_id, transcription_provider_name):
        #     err_msg = f"TranscriptionError: API key for {transcription_provider_name.capitalize()} not found or session expired."
        #     segment_result["error_details"].append(err_msg)
        #     await websocket_manager.send_personal_message({"status": "transcription_failed", "detail": err_msg}, user_id)
        # else:
        try:
            await websocket_manager.send_personal_message({"status": "transcription_started", "detail": f"Transcribing segment with {transcription_provider_name.capitalize()}..."}, user_id)
            transcription_text = None
            transcription_error = None
            if model_choice == ModelChoice.whissle:
                transcription_text, transcription_error = await transcribe_with_whissle_single(segment_path, user_id)
            elif model_choice == ModelChoice.gemini:
                transcription_text, transcription_error = await transcribe_with_gemini_single(segment_path, user_id)
            elif model_choice == ModelChoice.deepgram:
                transcription_text, transcription_error = await transcribe_with_deepgram_single(segment_path, user_id)
            else:
                transcription_error = "Invalid transcription model choice."
            if transcription_error:
                segment_result["error_details"].append(f"TranscriptionError: {transcription_error}")
                await websocket_manager.send_personal_message({"status": "transcription_failed", "detail": transcription_error}, user_id)
            elif transcription_text is None:
                segment_result["error_details"].append("TranscriptionError: Transcription returned None without an explicit error.")
                await websocket_manager.send_personal_message({"status": "transcription_failed", "detail": "Transcription returned no text."}, user_id)
            else:
                segment_result["transcription"] = transcription_text
                results["transcription"].append(transcription_text)
                await websocket_manager.send_personal_message({
                    "status": "transcription_complete",
                    "detail": "Segment transcription successful.",
                    "data": {"transcription": transcription_text[:100] + "..." if len(transcription_text) > 100 else transcription_text}
                }, user_id)
        except Exception as e:
            logger.error(f"User {user_id} - Transcription failed for segment {segment_path.name}: {e}", exc_info=True)
            err_msg = f"TranscriptionError: Unexpected error - {type(e).__name__}: {str(e)}"
            segment_result["error_details"].append(err_msg)
            await websocket_manager.send_personal_message({"status": "transcription_failed", "detail": err_msg}, user_id)

        # Annotations
        if transcription_text and requested_annotations:
            await websocket_manager.send_personal_message({"status": "annotation_started", "detail": f"Annotating segment {segment_path.name}..."}, user_id)

            # Load audio for age/gender/emotion
            audio_data: Optional[np.ndarray] = None
            sample_rate: Optional[int] = None
            if needs_local_models:
                await websocket_manager.send_personal_message({"status": "audio_loading", "detail": "Loading segment audio for age/gender/emotion..."}, user_id)
                audio_data, sample_rate, load_err = await asyncio.to_thread(load_audio, segment_path)
                if load_err:
                    segment_result["error_details"].append(f"AudioLoadError: {load_err}")
                    await websocket_manager.send_personal_message({"status": "audio_load_failed", "detail": load_err}, user_id)
                elif audio_data is None or sample_rate != TARGET_SAMPLE_RATE:
                    segment_result["error_details"].append("AudioLoadError: Audio load failed or sample rate mismatch.")
                    await websocket_manager.send_personal_message({"status": "audio_load_failed", "detail": "Audio load failed or sample rate mismatch."}, user_id)
                else:
                    await websocket_manager.send_personal_message({"status": "audio_load_complete", "detail": "Segment audio loaded successfully."}, user_id)

            # Age/Gender and Emotion Processing
            if audio_data is not None and sample_rate == TARGET_SAMPLE_RATE:
                tasks = []
                if any(a in ["age", "gender"] for a in requested_annotations) and 'age_gender_model' in globals():
                    await websocket_manager.send_personal_message({"status": "age_gender_started", "detail": "Processing age/gender for segment..."}, user_id)
                    tasks.append(asyncio.to_thread(predict_age_gender, audio_data, TARGET_SAMPLE_RATE))
                if "emotion" in requested_annotations and 'emotion_model' in globals():
                    await websocket_manager.send_personal_message({"status": "emotion_started", "detail": "Processing emotion for segment..."}, user_id)
                    tasks.append(asyncio.to_thread(predict_emotion, audio_data, TARGET_SAMPLE_RATE))

                if tasks:
                    try:
                        task_results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result in task_results:
                            if isinstance(result, Exception):
                                segment_result["error_details"].append(f"AnnotationError: {type(result).__name__}")
                                await websocket_manager.send_personal_message({"status": "annotation_failed", "detail": f"Annotation subtask error: {type(result).__name__}"}, user_id)
                                continue

                            if isinstance(result, tuple) and len(result) == 3:  # Age/Gender
                                age_pred, gender_idx, age_gender_err = result
                                if age_gender_err:
                                    segment_result["error_details"].append(f"AgeGenderError: {age_gender_err}")
                                    await websocket_manager.send_personal_message({"status": "age_gender_failed", "detail": age_gender_err}, user_id)
                                else:
                                    if "age" in requested_annotations and age_pred is not None:
                                        actual_age = round(age_pred, 1)
                                        age_brackets = [(18, "0-17"), (25, "18-24"), (35, "25-34"), (45, "35-44"), (55, "45-54"), (65, "55-64"), (float('inf'), "65+")]
                                        age_group = "Unknown"
                                        for threshold, bracket in age_brackets:
                                            if actual_age < threshold:
                                                age_group = bracket
                                                break
                                        segment_result["age_group"] = age_group
                                        results["age_group"].append(age_group)
                                    if "gender" in requested_annotations and gender_idx is not None:
                                        gender_str = "Unknown"
                                        if gender_idx == 1:
                                            gender_str = "Male"
                                        elif gender_idx == 0:
                                            gender_str = "Female"
                                        segment_result["gender"] = gender_str
                                        results["gender"].append(gender_str)
                                    await websocket_manager.send_personal_message({
                                        "status": "age_gender_complete",
                                        "data": {"age_group": segment_result["age_group"], "gender": segment_result["gender"]}
                                    }, user_id)
                            elif isinstance(result, tuple) and len(result) == 2:  # Emotion
                                emotion_label, emotion_err = result
                                if emotion_err:
                                    segment_result["error_details"].append(f"EmotionError: {emotion_err}")
                                    await websocket_manager.send_personal_message({"status": "emotion_failed", "detail": emotion_err}, user_id)
                                elif "emotion" in requested_annotations and emotion_label is not None:
                                    emotion = emotion_label.replace("_", " ").title() if emotion_label != "SHORT_AUDIO" else "Short Audio"
                                    segment_result["emotion"] = emotion
                                    results["emotion"].append(emotion)
                                    await websocket_manager.send_personal_message({"status": "emotion_complete", "data": {"emotion": emotion}}, user_id)
                    except Exception as e:
                        logger.error(f"User {user_id} - Error in A/G/E tasks for segment {segment_path.name}: {e}")
                        segment_result["error_details"].append(f"AnnotationError: {type(e).__name__}: {e}")
                        await websocket_manager.send_personal_message({"status": "annotation_failed", "detail": f"Annotation subtask error: {e}"}, user_id)

            # LLM Annotation
            if requires_annotation and transcription_text:
                if resolved_llm_model == LlmAnnotationModelChoice.gemini:
                    await websocket_manager.send_personal_message({"status": "gemini_annotation_started", "detail": "Starting Gemini entity/intent annotation for segment..."}, user_id)
                    if not GEMINI_AVAILABLE:
                        segment_result["error_details"].append("GeminiAnnotationError: Gemini SDK/api not available.")
                        await websocket_manager.send_personal_message({"status": "gemini_annotation_failed", "detail": "Gemini SDK not available."}, user_id)
                    # elif not get_user_api_key(user_id, "gemini"):
                    #     segment_result["error_details"].append("GeminiAnnotationError: Gemini API key not found or session expired.")
                    #     await websocket_manager.send_personal_message({"status": "gemini_annotation_failed", "detail": "Gemini API key not found or session expired."}, user_id)
                    else:
                        try:
                            tokens, tags, intent, gemini_err = await annotate_text_structured_with_gemini(
                                transcription_text,
                                custom_prompt=custom_prompt,
                                user_id=user_id
                            )
                            if gemini_err:
                                segment_result["error_details"].append(f"GeminiAnnotationError: {gemini_err}")
                                await websocket_manager.send_personal_message({"status": "gemini_annotation_failed", "detail": gemini_err}, user_id)
                                if "intent" in requested_annotations:
                                    segment_result["gemini_intent"] = "ANNOTATION_FAILED"
                            else:
                                # segment_result["prompt_used"] = custom_prompt if custom_prompt else "default_generated_prompt_behavior"
                                if "entity" in requested_annotations and tokens and tags:
                                    segment_result["bio_annotation_gemini"] = BioAnnotation(tokens=tokens, tags=tags).dict()
                                    results["bio_annotation_gemini"].append(segment_result["bio_annotation_gemini"])
                                if "intent" in requested_annotations and intent:
                                    segment_result["gemini_intent"] = intent
                                    results["gemini_intent"].append(intent)
                                await websocket_manager.send_personal_message({
                                    "status": "gemini_annotation_complete",
                                    "data": {
                                        "bio_annotation_gemini": segment_result["bio_annotation_gemini"],
                                        "gemini_intent": segment_result["gemini_intent"],
                                        # "prompt_used": segment_result["prompt_used"]
                                    }
                                }, user_id)
                        except Exception as e:
                            logger.error(f"User {user_id} - Gemini annotation failed for segment {segment_path.name}: {e}", exc_info=True)
                            segment_result["error_details"].append(f"GeminiAnnotationError: {type(e).__name__}: {str(e)}")
                            await websocket_manager.send_personal_message({"status": "gemini_annotation_failed", "detail": f"Gemini annotation error: {e}"}, user_id)
                            if "intent" in requested_annotations:
                                segment_result["gemini_intent"] = "ANNOTATION_FAILED"
                else:
                    # Unsupported LLM annotation model
                    error_msg = f"Annotation model '{resolved_llm_model.value}' is not supported; please use 'gemini' for now."
                    segment_result["error_details"].append(error_msg)
                    segment_result["annotation_model_error"] = error_msg
                    await websocket_manager.send_personal_message({"status": "annotation_failed", "detail": error_msg}, user_id)
                    if "intent" in requested_annotations:
                        segment_result["gemini_intent"] = "MODEL_NOT_SUPPORTED"

            elif requested_annotations and not transcription_text:
                segment_result["error_details"].append("AnnotationSkipped: Transcription failed or was empty.")
                await websocket_manager.send_personal_message({"status": "annotation_skipped", "detail": "Transcription failed or was empty, skipping annotations."}, user_id)

            await websocket_manager.send_personal_message({"status": "segment_annotation_complete", "detail": f"Annotations complete for segment {segment_path.name}."}, user_id)

        # Save Segment Result
        try:
            record = {
                "audio_filepath": str(segment_path),
                "original_gcs_path": original_gcs_path,
                "text": segment_result["transcription"],
                "duration": segment_result["duration"],
                "model_used_for_transcription": model_choice.value,
                "age_group": segment_result["age_group"],
                "gender": segment_result["gender"],
                "emotion": segment_result["emotion"],
                "bio_annotation_gemini": segment_result["bio_annotation_gemini"],
                "gemini_intent": segment_result["gemini_intent"],
                "annotation_model_error": segment_result.get("annotation_model_error"),
                # "prompt_used": segment_result["prompt_used"],
                "error": "; ".join(segment_result["error_details"]) if segment_result["error_details"] else None
            }
            record = {k: v for k, v in record.items() if v is not None}
            with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                json.dump(record, f_out, ensure_ascii=False)
                f_out.write('\n')
            await websocket_manager.send_personal_message({"status": "segment_result_saved", "detail": f"Segment result saved to {output_jsonl_path.name}"}, user_id)
            segment_results.append(segment_result)
        except Exception as e:
            logger.error(f"User {user_id} - Failed to save segment result for {segment_path.name}: {e}")
            segment_result["error_details"].append(f"SaveError: {str(e)}")
            await websocket_manager.send_personal_message({"status": "save_failed", "detail": f"Failed to save segment result: {str(e)}"}, user_id)

        # Aggregate errors
        if segment_result["error_details"]:
            results["error_details"].extend(segment_result["error_details"])

    # Finalize Results
    results["overall_error_summary"] = "; ".join(results["error_details"]) if results["error_details"] else None
    if results["overall_error_summary"]:
        await websocket_manager.send_personal_message({"status": "processing_failed", "detail": results["overall_error_summary"], "data": results}, user_id)
    else:
        await websocket_manager.send_personal_message({"status": "processing_complete", "detail": "All processing finished successfully.", "data": results}, user_id)

    return results
