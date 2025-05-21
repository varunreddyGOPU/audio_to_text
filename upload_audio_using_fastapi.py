import whisper
import torchaudio
import warnings
import torch
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

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

        # Transcribe the audio using the Whisper model
        result = model.transcribe(audio)

        # Return the transcribed text
        return JSONResponse(content={"transcription": result["text"]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

    finally:
        # Clean up: Delete the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)