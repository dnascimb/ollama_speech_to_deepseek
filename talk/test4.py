import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample

# --------------------------
# Load Whisper model
# --------------------------
model = WhisperModel("medium", device="cpu")  # or "small" if CPU is slow

# --------------------------
# Configuration
# --------------------------
chunk_duration = 2  # seconds per chunk
target_sr = 16000   # Whisper expects 16 kHz

# List devices and pick microphone
print("Available input devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"{i}: {dev['name']}")

mic_index = int(input("Select input device index: "))
dev = sd.query_devices(mic_index)
channels = min(1, dev['max_input_channels'])
print(f"Using {channels} channel(s), device default rate {dev['default_samplerate']}")

# --------------------------
# Streaming callback
# --------------------------
def record_chunk(duration):
    frames = int(duration * dev['default_samplerate'])
    audio = sd.rec(frames, samplerate=dev['default_samplerate'],
                   channels=channels, dtype='float32', device=mic_index)
    sd.wait()
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # Resample to 16 kHz
    audio_16k = resample(audio, int(len(audio) * target_sr / dev['default_samplerate']))
    # Normalize
    audio_16k /= np.max(np.abs(audio_16k) + 1e-9)
    return audio_16k

# --------------------------
# Main loop
# --------------------------
print("\n🎙️ Streaming speech-to-text. Press Ctrl+C to quit.\n")
try:
    while True:
        audio_chunk = record_chunk(chunk_duration)
        segments, _ = model.transcribe(audio_chunk, beam_size=5)
        text = " ".join([seg.text for seg in segments]).strip()
        if text:
            print(f"[Transcribed]: {text}")
except KeyboardInterrupt:
    print("\nExiting...")
