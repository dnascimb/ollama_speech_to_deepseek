import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# Load the model
model = WhisperModel("small", device="cpu")  # or "medium" if you have enough RAM

# Test with a WAV file
audio_file = "debug_recording_16k.wav"
data, samplerate = sf.read(audio_file)

# Ensure mono
if data.ndim > 1:
    data = np.mean(data, axis=1)

# Convert to float32
data = data.astype(np.float32)

# Transcribe
segments, _ = model.transcribe(data, beam_size=5, word_timestamps=False)

text = " ".join([seg.text for seg in segments]).strip()
print("Transcription:", text)
