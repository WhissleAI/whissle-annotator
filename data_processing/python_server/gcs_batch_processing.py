import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from gcs_utils import list_gcs_files, download_gcs_blob, parse_gcs_path
from websocket_utils import manager as websocket_manager
from gcs_single_file_processing import _process_single_downloaded_file
from config import ModelChoice, LlmAnnotationModelChoice


async def process_gcs_directory(
    user_id: str,
    gcs_dir_path: str,
    model_choice: ModelChoice,
    requested_annotations: Optional[List[str]],
    llm_annotation_model: Optional[LlmAnnotationModelChoice],
    custom_prompt: Optional[str],
    output_jsonl_path: Path,
    path_type: Optional[str] = None,
    segment_length_sec: Optional[float] = None,
    segment_overlap_sec: Optional[float] = None
) -> Dict[str, Any]:
    """
    Processes all .wav files in a GCS directory: downloads, transcribes/annotates, and saves results.
    """
    await websocket_manager.send_personal_message({"status": "gcs_dir_listing", "detail": f"Listing .wav files in {gcs_dir_path}"}, user_id)
    bucket_name, prefix = parse_gcs_path(gcs_dir_path)
    if not bucket_name or not prefix:
        await websocket_manager.send_personal_message({"status": "error", "detail": "Invalid GCS directory path."}, user_id)
        return {"error": "Invalid GCS directory path."}

    # List all .wav files in the directory
    wav_files = await asyncio.to_thread(list_gcs_files, bucket_name, prefix, ".wav")
    if not wav_files:
        await websocket_manager.send_personal_message({"status": "no_files", "detail": "No .wav files found in directory."}, user_id)
        return {"error": "No .wav files found in directory."}

    await websocket_manager.send_personal_message({"status": "gcs_dir_found", "detail": f"Found {len(wav_files)} .wav files."}, user_id)
    results = []
    errors = []
    for idx, blob_name in enumerate(wav_files):
        await websocket_manager.send_personal_message({"status": "gcs_file_processing", "detail": f"Processing file {idx+1}/{len(wav_files)}: {blob_name}"}, user_id)
        local_audio_path = await asyncio.to_thread(download_gcs_blob, bucket_name, blob_name)
        if not local_audio_path:
            errors.append(f"Failed to download {blob_name}")
            continue
        single_result = await _process_single_downloaded_file(
            local_audio_path,
            user_id,
            model_choice,
            requested_annotations,
            llm_annotation_model,
            custom_prompt,
            output_jsonl_path,
            f"gs://{bucket_name}/{blob_name}",
            segment_length_sec,
            segment_overlap_sec
        )
        results.append(single_result)
        try:
            if local_audio_path.exists():
                local_audio_path.unlink()
        except Exception:
            pass
    await websocket_manager.send_personal_message({"status": "gcs_dir_complete", "detail": f"Processed {len(results)} files. Errors: {len(errors)}"}, user_id)
    return {"processed_files": len(results), "errors": errors, "results": results}
