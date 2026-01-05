import argparse
import json
import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =========================
# DEFAULTS
# =========================
DEFAULT_INPUT = "sampleinput.jsonl"
DEFAULT_OUTPUT = "emotion_benchmark_results.jsonl"

# =========================
# API CONFIG
# =========================
WHISSLE_API_KEY = os.getenv("WHISSLE_API_KEY")
BS_API_KEY = os.getenv("BS_API_KEY")
LIGHTNING_API_KEY = os.getenv("LIGHTNING_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Endpoints (override via env if needed)
# Whissle uses auth_token as query param, not Bearer header
WHISSLE_URL = os.getenv("WHISSLE_EMOTION_URL", "https://api.whissle.ai/v1/conversation/STT")
BS_URL = os.getenv("BS_EMOTION_URL", "https://platform.behavioralsignals.com/v3/analyze")
LIGHTNING_URL = os.getenv("LIGHTNING_EMOTION_URL", "https://lightning.ai/lightning-ai/api/temp_01jmmjrkexkc4fprvbx374r6b0")
DEEPGRAM_URL = os.getenv("DEEPGRAM_SENTIMENT_URL", "https://api.deepgram.com/v1/listen?sentiment=true")

# Emotion tag pattern in text: EMOTION_SAD, EMOTION_HAP, EMOTION_NEU, etc.
EMOTION_PATTERN = re.compile(r"EMOTION_(\w+)")

# Map short codes to full labels
EMOTION_MAP = {
    "SAD": "sad",
    "HAP": "happy",
    "NEU": "neutral",
    "ANG": "angry",
    "FEA": "fear",
    "DIS": "disgust",
    "SUR": "surprise",
}


# =========================
# HELPERS
# =========================
def extract_ground_truth_emotion(text: str) -> str | None:
    """Extract emotion label from text like '... EMOTION_SAD ...'"""
    if not text:
        return None
    match = EMOTION_PATTERN.search(text)
    if match:
        code = match.group(1).upper()
        return EMOTION_MAP.get(code, code.lower())
    return None


def safe_json(response: requests.Response) -> dict:
    """Parse JSON or return error dict."""
    try:
        return response.json()
    except ValueError:
        return {
            "error": "non_json_response",
            "status_code": response.status_code,
            "text_preview": (response.text or "")[:300],
        }


def extract_emotion_label(payload: dict, service: str) -> str | None:
    """Best-effort extraction of emotion label from API response."""
    if not payload or payload.get("error"):
        return f"ERROR:{payload.get('error', 'unknown')}" if payload else "ERROR:empty"

    # Whissle-specific: emotion is embedded in transcript as EMOTION_XXX tag
    if service == "whissle":
        transcript = payload.get("transcript") or ""
        emotion = extract_ground_truth_emotion(transcript)
        if emotion:
            return emotion
        # Also check other possible fields
        if payload.get("emotion"):
            return str(payload["emotion"]).lower()

    # BehavioralSignals-specific: results array with task="emotion"
    if service == "behavioral_signals":
        results = payload.get("results") or []
        for result in results:
            if isinstance(result, dict) and result.get("task") == "emotion":
                # Use finalLabel if available
                final_label = result.get("finalLabel")
                if final_label:
                    return str(final_label).lower()
                # Otherwise get highest posterior from prediction array
                predictions = result.get("prediction") or []
                if predictions:
                    # Sort by posterior (descending) and get top label
                    try:
                        sorted_preds = sorted(
                            predictions,
                            key=lambda x: float(x.get("posterior", 0)),
                            reverse=True
                        )
                        if sorted_preds:
                            return str(sorted_preds[0].get("label", "")).lower()
                    except (ValueError, TypeError):
                        pass
        return None

    # Try common keys
    for key in ["emotion", "label", "predicted_emotion", "result", "sentiment"]:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()

    # Nested: predictions/results list
    for list_key in ["predictions", "results", "emotions", "data"]:
        items = payload.get(list_key)
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, dict):
                for k in ["emotion", "label", "name", "class"]:
                    v = first.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip().lower()
            elif isinstance(first, str):
                return first.strip().lower()
        elif isinstance(items, dict):
            # Dict mapping emotion->score, pick highest
            if items:
                top = max(items.items(), key=lambda x: float(x[1]) if x[1] else 0)
                return top[0].lower()

    return None


def extract_deepgram_sentiment(payload: dict) -> str | None:
    """Extract average sentiment from Deepgram response."""
    if not payload or payload.get("error"):
        return f"ERROR:{payload.get('error', 'unknown')}" if payload else "ERROR:empty"

    try:
        avg = payload.get("results", {}).get("sentiments", {}).get("average", {})
        sentiment = avg.get("sentiment")
        if sentiment:
            return sentiment.lower()
    except Exception:
        pass

    return None


# =========================
# API CALLS
# =========================
def call_whissle(audio_path: str) -> dict:
    """Call Whissle STT API with auth_token as query param, file as 'audio' field."""
    if not WHISSLE_API_KEY:
        return {"error": "missing_api_key"}
    
    # Whissle uses auth_token query param, not Authorization header
    url = f"{WHISSLE_URL}?auth_token={WHISSLE_API_KEY}"
    headers = {"Accept": "*/*"}
    
    try:
        with open(audio_path, "rb") as f:
            # Field name is 'audio', not 'file'
            files = {"audio": (Path(audio_path).name, f)}
            r = requests.post(url, headers=headers, files=files, timeout=120)
        return safe_json(r)
    except Exception as e:
        return {"error": str(e)}


def call_behavioral_signals(audio_path: str) -> dict:
    """
    BehavioralSignals API using official SDK:
    1. Upload audio -> get pid
    2. Poll for results using pid
    
    Response has results array with task="emotion" containing prediction labels.
    """
    if not BS_API_KEY:
        return {"error": "missing_api_key"}
    
    bs_client_id = os.getenv("BS_CLIENT_ID")
    if not bs_client_id:
        return {"error": "missing_client_id", "message": "Set BS_CLIENT_ID env var"}
    
    try:
        from behavioralsignals import Client
        
        client = Client(bs_client_id, BS_API_KEY)
        
        # Step 1: Upload audio
        upload_response = client.behavioral.upload_audio(file_path=audio_path)
        pid = upload_response.pid
        
        if not pid:
            return {"error": "no_pid_returned"}
        
        # Step 2: Poll for results
        import time
        max_attempts = 60  # 60 attempts x 2 seconds = 120 seconds max wait
        for attempt in range(max_attempts):
            try:
                result_response = client.behavioral.get_result(pid=pid)
                
                # Check if processing is complete
                if hasattr(result_response, 'results') and result_response.results:
                    # Convert to dict for consistent handling
                    return {
                        "pid": pid,
                        "results": [
                            {
                                "task": r.task if hasattr(r, 'task') else None,
                                "finalLabel": r.finalLabel if hasattr(r, 'finalLabel') else None,
                                "prediction": [
                                    {"label": p.label, "posterior": p.posterior}
                                    for p in (r.prediction if hasattr(r, 'prediction') and r.prediction else [])
                                ] if hasattr(r, 'prediction') else []
                            }
                            for r in result_response.results
                        ]
                    }
                elif hasattr(result_response, 'status') and result_response.status == 0:
                    # Still pending
                    time.sleep(2)
                    continue
                else:
                    time.sleep(2)
                    continue
                    
            except Exception as poll_error:
                # May get error while still processing
                time.sleep(2)
                continue
        
        return {"error": "timeout", "message": "Results not ready after 120 seconds"}
        
    except ImportError:
        return {"error": "sdk_not_installed", "message": "pip install behavioralsignals"}
    except Exception as e:
        return {"error": str(e)}


def call_lightning(audio_path: str) -> dict:
    if not LIGHTNING_API_KEY:
        return {"error": "missing_api_key"}
    headers = {"Authorization": f"Bearer {LIGHTNING_API_KEY}"}
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (Path(audio_path).name, f)}
            r = requests.post(LIGHTNING_URL, headers=headers, files=files, timeout=60)
        return safe_json(r)
    except Exception as e:
        return {"error": str(e)}


def call_deepgram(audio_path: str) -> dict:
    if not DEEPGRAM_API_KEY:
        return {"error": "missing_api_key"}
    # Detect content type from extension
    ext = Path(audio_path).suffix.lower()
    content_type_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }
    content_type = content_type_map.get(ext, "audio/mpeg")

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type,
    }
    try:
        with open(audio_path, "rb") as f:
            r = requests.post(DEEPGRAM_URL, headers=headers, data=f, timeout=120)
        return safe_json(r)
    except Exception as e:
        return {"error": str(e)}


# =========================
# MAIN
# =========================
def process_manifest(input_path: str, output_path: str):
    results = []

    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} entries in {input_path}")

    for idx, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  [!] Skipping invalid JSON at line {idx}: {e}")
            continue

        audio_path = entry.get("audio_filepath")
        text = entry.get("text", "")

        if not audio_path:
            print(f"  [!] Skipping line {idx}: no audio_filepath")
            continue

        if not Path(audio_path).exists():
            print(f"  [!] Skipping {audio_path}: file not found")
            continue

        print(f"[{idx}/{len(lines)}] Processing: {Path(audio_path).name}")

        # Extract ground truth
        ground_truth = extract_ground_truth_emotion(text)

        # Call APIs
        whissle_resp = call_whissle(audio_path)
        bs_resp = call_behavioral_signals(audio_path)
        lightning_resp = call_lightning(audio_path)
        deepgram_resp = call_deepgram(audio_path)

        # Extract labels
        whissle_emotion = extract_emotion_label(whissle_resp, "whissle")
        bs_emotion = extract_emotion_label(bs_resp, "behavioral_signals")
        lightning_emotion = extract_emotion_label(lightning_resp, "lightning")
        deepgram_sentiment = extract_deepgram_sentiment(deepgram_resp)

        result = {
            "audio_filepath": audio_path,
            "ground_truth_emotion": ground_truth,
            "whissle_emotion": whissle_emotion,
            "behavioralsignals_emotion": bs_emotion,
            "lightning_emotion": lightning_emotion,
            "deepgram_sentiment": deepgram_sentiment,
        }

        results.append(result)

        # Append to output file incrementally
        with open(output_path, "a") as out:
            out.write(json.dumps(result) + "\n")

        print(f"    GT={ground_truth} | Whissle={whissle_emotion} | BS={bs_emotion} | Lightning={lightning_emotion} | DG={deepgram_sentiment}")

    print(f"\nDone! Wrote {len(results)} results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark emotion detection APIs against ground truth")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input JSONL manifest")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output JSONL results")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # Clear output file if exists
    if Path(output_path).exists():
        Path(output_path).unlink()

    process_manifest(input_path, output_path)


if __name__ == "__main__":
    main()
