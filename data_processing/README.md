# Data Processing Server and Jupyter Notebook Usage

This document outlines how to use the Python server for data processing, specifically through the provided Jupyter notebooks for handling audio files stored in Google Cloud Storage (GCS).

## Overview

The setup consists of:

- A **Python FastAPI server** (`data_processing/python_server`) that exposes endpoints for processing audio files.
- **Jupyter notebooks** (`data_processing/jupiter`) that provide a client interface to interact with the server for tasks like transcription and annotation.

The primary workflow involves sending requests from a Jupyter notebook to the FastAPI server, pointing to audio data in GCS.

## End-to-End Workflow: Annotating Soccer Audio and Uploading to Hugging Face

This section provides a complete step-by-step guide for taking a GCS folder of soccer audio, annotating it using the server, and uploading the final dataset to Hugging Face.

### **Part 1: Annotating Soccer Audio Data from GCS**

#### **Step 1: Environment Setup**

Before running the server, ensure your environment is correctly configured.

1.  **Install Dependencies:**
    Navigate to the Python server directory and install the required packages.

    ```bash
    cd D:\z-whissle\meta-asr\data_processing\python_server
    pip install -r requirements.txt
    ```

2.  **Set API Keys:**
    The server requires a Google API key for transcription and annotation with Gemini. Make sure your `.env` file (located at `D:\z-whissle\meta-asr\.env`) contains your key:
    ```env
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

#### **Step 2: Start the FastAPI Server**

This server will handle the heavy lifting of downloading, transcribing, and annotating the audio files.

1.  **Navigate to the server directory:**

    ```bash
    cd D:\z-whissle\meta-asr\data_processing\python_server
    ```

2.  **Run the server:**
    ```bash
    python main.py
    ```
    The server will start on `http://localhost:8000`. Keep this terminal window open.

#### **Step 3: Configure and Run the Jupyter Notebook for Annotation**

The notebook acts as a client to send requests to your running server.

1.  **Open the Notebook:**
    Open the Jupyter Notebook located at:
    `D:\z-whissle\meta-asr\data_processing\jupiter\soccer_data_directory_finetuning.ipynb`

2.  **Review the Annotation Prompt:**
    The **first code cell** defines a detailed `soccer_prompt`. This prompt is critical as it instructs the Gemini model on how to perform the soccer-specific annotation, including the entity types (like `PLAYER_NAME`, `TEAM_NAME`, `MATCH_ACTION`) and intent types (`LIVE_COMMENTARY`, `GOAL_ANNOUNCEMENT`).

3.  **Configure the API Request:**
    In the **second code cell**, you will find the `payload` dictionary. Verify the following parameters:

    - `gcs_path`: Set this to the GCS path of your folder containing the soccer audio files. The current example uses `gs://stream2action-audio/youtube-videos/soccer_data`.
    - `output_jsonl_path`: This can be an absolute path (e.g., `D:/z-whissle/meta-asr/data_processing/output`) or a relative path (e.g., `output`). If relative, it will be resolved relative to `DEFAULT_OUTPUT_DIR` or `data_processing/outputs` by default.
    - `annotations`: Ensure this list includes `"entity"` and `"intent"` to get the detailed soccer annotations.
    - `prompt`: (Optional) You can provide a custom prompt, or omit it to use the default annotation prompt. The notebook example shows how to load a custom soccer-specific prompt from a file.

4.  **Execute the Notebook Cells:**
    Run the cells in the notebook sequentially. The second cell will send the request to the running FastAPI server. The server will then begin processing the audio files from the GCS folder. This process may take some time depending on the number and length of the audio files.

#### **Step 4: Review the Annotated Output**

Once the server has finished, the annotated data will be available in the output directory you specified.

1.  **Locate the Output:**
    Navigate to the directory you set in `output_jsonl_path`. Inside, you will find subdirectories for each processed audio file. Each subdirectory contains:

    - A `segments` folder with the chunked audio clips.
    - An `annotations.jsonl` file with the detailed annotation data for each segment.

2.  **Understand the Output Format:**
    Each line in the `annotations.jsonl` file is a JSON object containing the transcription and the soccer-specific annotations as requested in the prompt, including:
    - `tokens`: A list of words from the transcription.
    - `tags`: A list of BIO tags (e.g., `B-PLAYER_NAME`, `I-TEAM_NAME`, `B-LIVE_COMMENTARY`) corresponding to each token.
    - `intent`: The overall intent of the sentence (e.g., `LIVE_COMMENTARY`).

### **Part 2: Preparing and Uploading to Hugging Face**

After annotating your data, the same notebook helps you prepare and upload it.

#### **Step 1: Aggregate Processed Data into a Dataset**

1.  **Execute the aggregation cell:**
    - In the same notebook, run the cell under the markdown comment `"merge whole stuff into one dataset, before pushing into hf"`.
    - This script reads the processed files from the directory specified in `output_jsonl_path` (e.g., `D:\z-whissle\meta-asr\data_processing\hello`).
    - It then creates a final, structured dataset in `D:\z-whissle\meta-asr\final_datasets\combined_dataset`, which includes an `audio` folder and an `annotations.jsonl` file.

#### **Step 2: Push the Dataset to Hugging Face**

1.  **Set your Hugging Face Token:**

    - In the last code cell of the notebook, replace the placeholder text `"user your own"` with your actual Hugging Face access token.

    ```python
    # Change this line
    HF_TOKEN = "hf_YourAccessTokenHere"
    ```

2.  **Execute the upload cell:**
    - Run this final cell. It will use your token to authenticate and then upload the contents of the `D:\z-whissle\meta-asr\final_datasets\combined_dataset` directory to a new or existing repository on your Hugging Face profile named `audio_combined_dataset`.

## General API Usage Examples

The Jupyter notebooks in `data_processing/jupiter/` are designed for processing audio data from GCS.

**For Postman and cURL usage**, see the comprehensive API documentation: [`data_processing/python_server/API_EXAMPLES.md`](python_server/API_EXAMPLES.md)

This documentation includes:

- Complete request/response examples
- cURL commands ready to copy-paste
- Postman collection setup guide
- Troubleshooting tips
- Information about default prompts and relative path handling

### 1. Processing a Single GCS Audio File

The `soccer_single_file_finetuning.ipynb` notebook demonstrates how to process a single audio file from a GCS bucket.

**Endpoint**: `POST /process_gcs_file/`

**Example Request Payload**:
The notebook sends a POST request to the server with a JSON payload. The key fields are:

- `user_id`: A unique identifier for the user making the request.
- `gcs_path`: The full `gs://` path to the single audio file.
- `model_choice`: The transcription model to use (e.g., "gemini").
- `output_jsonl_path`: An absolute or relative path where the output will be saved. If relative, it will be resolved relative to `DEFAULT_OUTPUT_DIR` or `data_processing/outputs` by default.
- `annotations`: A list of annotations to perform (e.g., "entity", "intent").
- `prompt`: (Optional) A custom prompt string to guide the annotation model. If not provided and entity/intent annotations are requested, a default prompt will be used automatically.

**Jupyter Notebook Snippet**:

```python
import requests
import json

# Load a custom prompt from a file
with open("soccer_prompt.txt") as f:
    custom_prompt = f.read()

# Build the request payload
payload = {
    "user_id": "user_123",
    "gcs_path": "gs://stream2action-audio/youtube-videos/soccer_data/England_v_Italy_-_Watch_the_full_2012_penalty_shoot-out_16k.wav",
    "model_choice": "gemini",
    "output_jsonl_path": "/home/dchauhan/workspace/meta-asr/data_processing/hello",
    "annotations": ["age", "gender", "emotion", "entity", "intent"],
    "prompt": custom_prompt
}

# Send the request to the server
url = "http://localhost:8000/process_gcs_file/"
response = requests.post(url, json=payload)

# Print the server's response
print(json.dumps(response.json(), indent=2))
```

### 2. Processing a GCS Directory

The `soccer_data_directory_finetuning.ipynb` notebook shows how to process all audio files within a specified GCS directory (folder).

**Endpoint**: `POST /process_gcs_directory/`

**Example Request Payload**:
The payload is identical to the single file request, but the `gcs_path` points to a directory instead of a file. The server will find and process all compatible audio files in that directory.

**Jupyter Notebook Snippet**:

```python
import requests
import json

# Load a custom prompt from a file
with open("soccer_prompt.txt") as f:
    custom_prompt = f.read()

# Build the request payload
payload = {
    "user_id": "user_123",
    "gcs_path": "gs://stream2action-audio/youtube-videos/soccer_data", # Path to the GCS directory
    "model_choice": "gemini",
    "output_jsonl_path": "/home/dchauhan/workspace/meta-asr/data_processing/hello",
    "annotations": ["age", "gender", "emotion", "entity", "intent"],
    "prompt": custom_prompt
}

# Send the request to the server
url = "http://localhost:8000/process_gcs_directory/"
response = requests.post(url, json=payload)

# Print the server's response
print(json.dumps(response.json(), indent=2))
```
