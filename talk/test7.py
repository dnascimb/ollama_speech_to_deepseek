import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample
import soundfile as sf
import os
import argparse

# --------------------------
# Configuration
# --------------------------
MODEL_NAME = "small"       # small = faster, medium = more accurate
DEVICE = "cpu"             # or "mps" for Metal GPU on Mac
CHUNK_DURATION = 1         # seconds per chunk
TARGET_SR = 16000          # Whisper expects 16 kHz
OUTPUT_WAV = "session.wav" # continuous recording

# --------------------------
# Parse arguments
# --------------------------
parser = argparse.ArgumentParser(description="Realtime STT or transcribe a WAV file with faster-whisper.")
parser.add_argument("--file", type=str, help="Path to WAV file to transcribe instead of using microphone.")
args = parser.parse_args()

# --------------------------
# Load Whisper model
# --------------------------
print("Loading Whisper model...")
model = WhisperModel(MODEL_NAME, device=DEVICE)
print("Model loaded.\n")

# --------------------------
# WAV file mode
# --------------------------
if args.file:
    print(f"📂 Transcribing from file: {args.file}")
    audio, sr = sf.read(args.file)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # convert to mono

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        audio = resample(audio, int(len(audio) * TARGET_SR / sr))
        sr = TARGET_SR

    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio) + 1e-9)

    segments, _ = model.transcribe(audio, beam_size=1, language="en")
    text = " ".join([seg.text for seg in segments]).strip()
    print(f"[Transcription]: {text}")
    exit(0)

# --------------------------
# Microphone mode
# --------------------------
print("Available input devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"{i}: {dev['name']}")

mic_index = int(input("Select input device index: "))
dev = sd.query_devices(mic_index)
channels = min(1, dev['max_input_channels'])
print(f"Using {channels} channel(s), device default rate {dev['default_samplerate']} Hz\n")

if os.path.exists(OUTPUT_WAV):
    os.remove(OUTPUT_WAV)

wav_file = sf.SoundFile(OUTPUT_WAV, mode='w', samplerate=TARGET_SR,
                        channels=1, subtype='PCM_16')

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

print("🎙️ Streaming STT. Press Ctrl+C to quit.\n")
try:
    while True:
        chunk = record_chunk(CHUNK_DURATION)
        wav_file.write(chunk)
        wav_file.flush()
        segments, _ = model.transcribe(chunk, beam_size=1, language="en")
        text = " ".join([seg.text for seg in segments]).strip()
        if text:
            print(f"[Transcribed]: {text}")

except KeyboardInterrupt:
    wav_file.close()
    print("\nExiting... recording saved to", OUTPUT_WAV)
