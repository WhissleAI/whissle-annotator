# applications/audio_utils.py
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import resampy
import requests
import time
from config import AUDIO_EXTENSIONS, TARGET_SAMPLE_RATE, logger, PYANNOTEAI_API_KEY, SpeakerSegment
from typing import Tuple, Optional, List, Dict, Any
from fastapi import HTTPException
from gcs_utils import parse_gcs_path
from pydub import AudioSegment # Add pydub import

def validate_paths(dir_path_str: str, output_path_str: str) -> Tuple[Path, Path]:
    dir_path = Path(dir_path_str)
    output_jsonl_path = Path(output_path_str)
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {dir_path_str}")
    if not output_jsonl_path.parent.is_dir():
        raise HTTPException(status_code=400, detail=f"Output directory does not exist: {output_jsonl_path.parent}")
    return dir_path, output_jsonl_path

def discover_audio_files(directory_path: Path) -> List[Path]:
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(directory_path.glob(f"*{ext}"))
        audio_files.extend(directory_path.glob(f"*{ext.upper()}"))
    audio_files.sort()
    logger.info(f"Discovered {len(audio_files)} audio files in {directory_path}")
    return audio_files

def load_audio(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
            sr = target_sr
        return audio, sr, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=False)
        return None, None, f"Failed to load audio: {type(e).__name__}"

def get_audio_duration(audio_path: Path) -> Optional[float]:
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        try:
            duration = librosa.get_duration(path=str(audio_path))
            return duration
        except Exception as le:
            logger.error(f"Failed to get duration for {audio_path.name}: {le}", exc_info=False)
            return None

# def trim_audio(audio_path: Path, segment_length_ms: int, output_dir: Path) -> List[Path]:
#     """
#     Trims an audio file into segments of a specified length.

#     Args:
#         audio_path: Path to the input audio file.
#         segment_length_ms: Desired length of each segment in milliseconds.
#         output_dir: Directory to save the trimmed audio segments.

#     Returns:
#         A list of paths to the trimmed audio segments.
#     """
#     try:
#         audio = AudioSegment.from_file(audio_path)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         trimmed_files = []
#         # Ensure segment_length_ms is an integer for slicing
#         step = int(segment_length_ms)
#         for i, chunk in enumerate(audio[::step]):
#             trimmed_file_path = output_dir / f"{audio_path.stem}_segment_{i}{audio_path.suffix}"
#             chunk.export(trimmed_file_path, format=audio_path.suffix[1:])
#             trimmed_files.append(trimmed_file_path)
#         return trimmed_files
#     except Exception as e:
#         logger.error(f"Error trimming audio file {audio_path}: {e}", exc_info=True)
#         return []


def trim_audio(audio_path: Path, segment_length_ms: int, output_dir: Path, overlap_ms: Optional[int] = None) -> List[Path]:
    """
    Trims an audio file into segments of a specified length with overlap.

    Args:
        audio_path: Path to the input audio file.
        segment_length_ms: Desired length of each segment in milliseconds (e.g., 30000 for 30 seconds).
        output_dir: Directory to save the trimmed audio segments.
        overlap_ms: Overlap between segments in milliseconds. Defaults to 10000 (10 seconds) if not provided.

    Returns:
        A list of paths to the trimmed audio segments.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        audio_duration_ms = len(audio)  # Duration in milliseconds
        segment_length_ms = int(segment_length_ms)
        overlap_ms = int(overlap_ms) if overlap_ms is not None else 10000  # Default 10 seconds overlap

        output_dir.mkdir(parents=True, exist_ok=True)
        trimmed_files = []

        # If audio is <= segment length, save it as is
        if audio_duration_ms <= segment_length_ms:
            trimmed_file_path = output_dir / f"{audio_path.stem}_segment_0{audio_path.suffix}"
            audio.export(trimmed_file_path, format=audio_path.suffix[1:])
            trimmed_files.append(trimmed_file_path)
            return trimmed_files

        # Calculate step size (segment length minus overlap)
        step_ms = segment_length_ms - overlap_ms
        if step_ms <= 0:
            logger.warning(f"Overlap ({overlap_ms}ms) is >= segment length ({segment_length_ms}ms). Using step size of 1ms.")
            step_ms = 1
        
        num_segments = max(1, (audio_duration_ms - segment_length_ms) // step_ms + 1)

        for i in range(num_segments):
            start_ms = i * step_ms
            end_ms = min(start_ms + segment_length_ms, audio_duration_ms)
            chunk = audio[start_ms:end_ms]
            trimmed_file_path = output_dir / f"{audio_path.stem}_segment_{i}{audio_path.suffix}"
            chunk.export(trimmed_file_path, format=audio_path.suffix[1:])
            trimmed_files.append(trimmed_file_path)

        # Handle remainder if audio duration is not perfectly divisible
        if audio_duration_ms % step_ms > 0 and audio_duration_ms > segment_length_ms:
            start_ms = audio_duration_ms - segment_length_ms
            chunk = audio[start_ms:audio_duration_ms]
            trimmed_file_path = output_dir / f"{audio_path.stem}_segment_{num_segments}{audio_path.suffix}"
            chunk.export(trimmed_file_path, format=audio_path.suffix[1:])
            trimmed_files.append(trimmed_file_path)

        logger.info(f"Trimmed {audio_path.name} into {len(trimmed_files)} segments (segment: {segment_length_ms}ms, overlap: {overlap_ms}ms)")
        return trimmed_files
    except Exception as e:
        logger.error(f"Error trimming audio file {audio_path}: {e}", exc_info=True)
        return []


def gcs_path_to_public_url(gcs_path: str) -> Optional[str]:
    """Return a storage.googleapis.com URL for a GCS path if we can parse it."""
    bucket, blob = parse_gcs_path(gcs_path)
    if not bucket or not blob:
        return None
    return f"https://storage.googleapis.com/{bucket}/{blob}"


def perform_pyannote_diarization(
    audio_url: str,
    api_key: Optional[str] = None,
    polling_interval: int = 10,
    timeout_seconds: int = 600
) -> Tuple[Optional[List[SpeakerSegment]], Optional[str]]:
    """
    Creates a diarization job using pyannoteAI precision-2 and polls until completion.
    Returns a list of SpeakerSegment or an error message.
    """
    api_key = api_key or PYANNOTEAI_API_KEY
    if not api_key:
        return None, "PYANNOTEAI_API_KEY not configured"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"url": audio_url}

    try:
        response = requests.post("https://api.pyannote.ai/v1/diarize", headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        err = f"Diarization request failed: {type(exc).__name__}: {str(exc)}"
        logger.error(err, exc_info=True)
        return None, err

    if response.status_code != 200:
        err = f"Diarization API responded with {response.status_code}: {response.text}"
        logger.error(err)
        return None, err

    job_data = response.json()
    job_id = job_data.get("jobId")
    if not job_id:
        err = "Diarization API returned no jobId."
        logger.error(err)
        return None, err

    logger.info(f"pyannoteAI diarization job created: {job_id}")
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            status_resp = requests.get(f"https://api.pyannote.ai/v1/jobs/{job_id}", headers=headers, timeout=30)
        except requests.RequestException as exc:
            err = f"Failed to poll diarization job: {type(exc).__name__}: {str(exc)}"
            logger.error(err, exc_info=True)
            return None, err

        if status_resp.status_code != 200:
            err = f"Diarization status check failed: {status_resp.status_code} - {status_resp.text}"
            logger.error(err)
            return None, err

        payload = status_resp.json()
        logger.info("\u001b[32mRaw diarization payload for job %s: %s\u001b[0m", job_id, payload)
        status = payload.get("status")
        if status == "succeeded":
            output_data = payload.get("output", {}) or {}
            diarization_segments = output_data.get("diarization")
            segments = output_data.get("segments", []) or diarization_segments or []
            speaker_segments = []
            for segment in segments:
                start = segment.get("start")
                end = segment.get("end")
                speaker = segment.get("speaker", "SPEAKER_00")
                if start is None or end is None:
                    continue
                speaker_segments.append(SpeakerSegment(start=float(start), end=float(end), speaker=speaker))
            logger.info(f"pyannoteAI diarization completed with {len(speaker_segments)} segments.")
            return speaker_segments, None
        if status in ("failed", "canceled"):
            err = f"Diarization job {status}: {payload.get('error')}"
            logger.error(err)
            return None, err

        logger.info(f"Diarization job {job_id} is {status}, waiting {polling_interval}s")
        time.sleep(polling_interval)

    err = f"Diarization job timed out after {timeout_seconds} seconds."
    logger.error(err)
    return None, err


def segment_audio_by_speakers(
    audio_path: Path,
    speaker_segments: List[SpeakerSegment],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Export audio chunks corresponding to speaker segments. Returns a list of dicts describing each chunk.
    """
    segments_info: List[Dict[str, Any]] = []
    try:
        audio = AudioSegment.from_file(audio_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, segment in enumerate(speaker_segments):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            start_ms = max(0, start_ms)
            end_ms = min(len(audio), end_ms)
            if start_ms >= end_ms:
                continue

            chunk = audio[start_ms:end_ms]
            safe_speaker = "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in segment.speaker)
            segment_path = output_dir / f"{audio_path.stem}_speaker_{safe_speaker}_seg_{idx}{audio_path.suffix}"
            chunk.export(segment_path, format=audio_path.suffix[1:])

            segments_info.append({
                "path": segment_path,
                "speaker": segment.speaker,
                "start": segment.start,
                "end": segment.end,
                "duration": (segment.end - segment.start)
            })
        logger.info(f"Split {audio_path.name} into {len(segments_info)} speaker segments.")
        logger.info("\u001b[32mExported speaker chunks: %s\u001b[0m", [info["speaker"] for info in segments_info])
    except Exception as exc:
        logger.error(f"Failed to split audio by speakers: {exc}", exc_info=True)
    return segments_info