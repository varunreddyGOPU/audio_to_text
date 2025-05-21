import whisper
import torchaudio
import warnings
import torch
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*FP16 is not supported on CPU.*")

# Initialize FastAPI app
app = FastAPI()

# Load the Whisper model
model = whisper.load_model("base")

# Move the model to GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")
else:
    print("CUDA is not available. Using CPU.")
model.eval()

# Define a Pydantic model for the response
class TranscriptionResponse(BaseModel):
    text: str
    timestamp: float  # Timestamp in seconds

# Define the upload and transcription endpoint
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Check if the uploaded file is an audio file
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file.")

    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Load the audio file using torchaudio
        waveform, sample_rate = torchaudio.load(temp_file_path)

        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:  # Check if the audio has multiple channels
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample the audio to 16000 Hz (required by Whisper)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

        # Convert the waveform to a numpy array
        audio = waveform.squeeze().numpy()

        # Define a generator function to process the audio in chunks
        def process_audio_in_chunks(audio, chunk_size=30):
            """
            Process the audio in chunks and yield transcription results in real-time.
            """
            total_duration = len(audio) / 16000  # Total duration in seconds
            chunk_samples = chunk_size * 16000  # Number of samples per chunk

            for start in range(0, len(audio), chunk_samples):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]

                # Transcribe the chunk
                result = model.transcribe(chunk)

                # Calculate the timestamp for the chunk
                timestamp = start / 16000

                # Yield the transcription result
                yield TranscriptionResponse(text=result["text"], timestamp=timestamp).json() + "\n"

        # Return the transcription results as a streaming response
        return StreamingResponse(process_audio_in_chunks(audio), media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

    finally:
        # Clean up: Delete the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)