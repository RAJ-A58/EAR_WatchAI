import wave
import math
import struct
import os

def create_tone(filename, frequency, duration_sec, volume=0.5):
    """
    Generates a standard WAV file with a sine wave tone.
    """
    sample_rate = 44100
    n_samples = int(sample_rate * duration_sec)
    
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"Generating {filename}...")
    
    with wave.open(filename, 'w') as wav_file:
        # Set parameters: 1 channel (mono), 2 bytes size (16-bit), 44100 Hz frame rate
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        # Write frames
        for i in range(n_samples):
            # Generate sine wave value (-1.0 to 1.0)
            t = float(i) / sample_rate
            value = math.sin(2.0 * math.pi * frequency * t)
            
            # Scale to 16-bit integer range (-32767 to 32767)
            sample = int(value * 32767.0 * volume)
            
            # Pack as little-endian 16-bit signed integer
            data = struct.pack('<h', sample)
            wav_file.writeframes(data)
            
    print(f"✓ Created {filename}")

if __name__ == "__main__":
    # Create assets folder if it doesn't exist
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # 1. Generate 'alarm.wav' (High pitch, fast beep style - 1000Hz)
    create_tone(os.path.join(assets_dir, "alarm.wav"), frequency=1000, duration_sec=1.0)

    # 2. Generate 'warning.wav' (Lower pitch, gentler - 400Hz)
    create_tone(os.path.join(assets_dir, "warning.wav"), frequency=400, duration_sec=0.5)

    print("\nDone! You can now run main.py")