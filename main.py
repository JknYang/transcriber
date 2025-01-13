import whisper
import pyaudio
import numpy as np
import os
from datetime import datetime

def record_audio_to_buffer(chunk_size=1024, rate=16000, overlap = .5):
    """Record audio from the microphone and yield chunks."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,channels=1,rate=rate,input=True,frames_per_buffer=chunk_size)

    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0 
    except KeyboardInterrupt:
        print("\nStopped recording.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def live_transcribe():
    model = whisper.load_model("base") 
    audio_buffer = []
    buffer_duration = 15  # Transcribe every 15 seconds of audio
    sample_rate = 16000  

    output_folder = "Transcriptions"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_folder, f"transcription_{current_time}.txt")

    print(f"Transcriptions will be saved to: {output_file}")

    with open(output_file, "w") as f:
        f.write(f"Live Transcriptions - {current_time}:\n\n")
    for audio_chunk in record_audio_to_buffer():
        audio_buffer.extend(audio_chunk)
        if len(audio_buffer) > sample_rate * buffer_duration:
            print("Transcribing...")
            audio_data = np.array(audio_buffer, dtype=np.float32)
            result = model.transcribe(audio_data, fp16=False) 
            transcription = result['text']
            print(f"Transcription: {transcription}")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_file, "a") as f:
                f.write(f"[{timestamp}] {transcription}\n")

            audio_buffer = []

if __name__ == "__main__":
    live_transcribe()
