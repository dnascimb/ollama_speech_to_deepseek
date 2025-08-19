import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample
import soundfile as sf
import os

# --------------------------
# Configuration
# --------------------------
MODEL_NAME = "small"       # "small" = fast, "medium" = more accurate
DEVICE = "cpu"             # or "mps" for Mac M1/M2 GPU
CHUNK_DURATION = 2         # seconds per chunk
TARGET_SR = 16000          # Whisper expects 16 kHz
OUTPUT_WAV = "session.wav" # continuous recording

# --------------------------
# Load Whisper model
# --------------------------
model = WhisperModel(MODEL_NAME, device=DEVICE)

# --------------------------
# Select microphone
# --------------------------
print("Available input devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"{i}: {dev['name']}")

mic_index = int(input("Select input device index: "))
dev = sd.query_devices(mic_index)
channels = min(1, dev['max_input_channels'])
print(f"Using {channels} channel(s), device default rate {dev['default_samplerate']} Hz")

# --------------------------
# Prepare WAV file for append
# --------------------------
if os.path.exists(OUTPUT_WAV):
    os.remove(OUTPUT_WAV)

# Open SoundFile in append mode
wav_file = sf.SoundFile(OUTPUT_WAV, mode='w', samplerate=TARGET_SR,
                        channels=1, subtype='PCM_16')

# --------------------------
# Record & transcribe chunk
# --------------------------
def record_chunk(duration):
    frames = int(duration * dev['default_samplerate'])
    audio = sd.rec(frames, samplerate=dev['default_samplerate'],
                   channels=channels, dtype='float32', device=mic_index)
    sd.wait()
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio_16k = resample(audio, int(len(audio) * TARGET_SR / dev['default_samplerate']))
    audio_16k /= np.max(np.abs(audio_16k) + 1e-9)
    return audio_16k

# --------------------------
# Main loop
# --------------------------
print("\n🎙️ Streaming STT. Press Ctrl+C to quit.\n")
try:
    while True:
        chunk = record_chunk(CHUNK_DURATION)
        
        # Append to WAV file
        wav_file.write(chunk)
        wav_file.flush()  # ensure it's written to disk

        # Transcribe with greedy decoding (beam_size=1) for speed
        segments, _ = model.transcribe(chunk, beam_size=1, language="en")
        text = " ".join([seg.text for seg in segments]).strip()
        if text:
            print(f"[Transcribed]: {text}")

except KeyboardInterrupt:
    wav_file.close()
    print("\nExiting... recording saved to", OUTPUT_WAV)
