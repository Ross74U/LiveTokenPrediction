import pyaudio
import soundfile as sf
import numpy as np
import time
from scipy import signal

class AudioFileStream:
    def __init__(self, filename, chunk_size=1024, target_sample_rate=44100):
        self.chunk_size = chunk_size
        self.filename = filename
        self.target_sample_rate = target_sample_rate
        
        try:
            # Open the audio file using soundfile
            self.audio_file = sf.SoundFile(filename)
            
            # Get file properties
            self.channels = self.audio_file.channels
            self.original_sample_rate = self.audio_file.samplerate
            self.subtype = self.audio_file.subtype
            
            # Calculate resampling ratio
            self.resampling_ratio = self.target_sample_rate / self.original_sample_rate
            
            # Map soundfile format to PyAudio format
            self.format = self._get_pyaudio_format(self.subtype)
            
            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.target_sample_rate,  # Use target sample rate
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_playing = False
            
        except Exception as e:
            print(f"Error opening audio file: {e}")
            raise

    def _get_pyaudio_format(self, subtype):
        format_map = {
            'PCM_16': pyaudio.paInt16,
            'PCM_24': pyaudio.paInt24,
            'PCM_32': pyaudio.paInt32,
            'FLOAT': pyaudio.paFloat32,
            'FLOAT32': pyaudio.paFloat32,
            'FLOAT64': pyaudio.paFloat32,
        }
        return format_map.get(subtype, pyaudio.paFloat32)

    def resample(self, audio_data):
        """Resample audio data to target sample rate"""
        if self.original_sample_rate == self.target_sample_rate:
            return audio_data
        
        # Calculate number of samples for target rate
        target_length = int(len(audio_data) * self.resampling_ratio)
        
        # Resample using scipy.signal.resample
        resampled = signal.resample(audio_data, target_length)
        
        return resampled
    
    def read(self, size):
        # Calculate frames needed based on size and format
        bytes_per_sample = pyaudio.get_sample_size(self.format)
        
        # Adjust frames needed based on resampling ratio
        original_frames_needed = int(size / (bytes_per_sample * self.channels * self.resampling_ratio))
        
        # Read frames from file
        frames = self.audio_file.read(original_frames_needed)
        
        # If we've reached the end of the file, start over
        if len(frames) < original_frames_needed:
            self.audio_file.seek(0)
            remaining_frames = original_frames_needed - len(frames)
            frames = np.concatenate([frames, self.audio_file.read(remaining_frames)])
        
        # Resample the audio data
        resampled_frames = self.resample(frames)
        
        # Convert to bytes based on format
        if self.format == pyaudio.paFloat32:
            return resampled_frames.astype(np.float32).tobytes()
        elif self.format == pyaudio.paInt16:
            return resampled_frames.astype(np.int16).tobytes()
        elif self.format == pyaudio.paInt24:
            temp = resampled_frames.astype(np.int32)
            return temp.tobytes()[:-(temp.size//3)]
        elif self.format == pyaudio.paInt32:
            return resampled_frames.astype(np.int32).tobytes()
    
    def start_stream(self):
        self.is_playing = True
        
    def stop_stream(self):
        self.is_playing = False
        
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio_file.close()
        self.p.terminate()

def main():
    # Create audio file stream with specific target sample rate
    audio_stream = AudioFileStream('your_audio_file.wav', 
                                 chunk_size=1024,
                                 target_sample_rate=16000)  # Example: downsample to 16kHz
    
    try:
        print("Starting stream...")
        print(f"File format: {audio_stream.subtype}")
        print(f"Channels: {audio_stream.channels}")
        print(f"Original sample rate: {audio_stream.original_sample_rate}")
        print(f"Target sample rate: {audio_stream.target_sample_rate}")
        
        audio_stream.start_stream()
        
        # Calculate chunk duration for real-time simulation
        bytes_per_sample = pyaudio.get_sample_size(audio_stream.format)
        chunk_duration = audio_stream.chunk_size / (audio_stream.target_sample_rate * audio_stream.channels * bytes_per_sample)
        
        while audio_stream.is_playing:
            start_time = time.time()
            
            # Read from the stream
            data = audio_stream.read(audio_stream.chunk_size)
            
            # Convert to numpy array for processing
            if audio_stream.format == pyaudio.paFloat32:
                audio_data = np.frombuffer(data, dtype=np.float32)
            else:
                audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Your processing code here...
            
            # Maintain real-time timing
            processing_time = time.time() - start_time
            if processing_time < chunk_duration:
                time.sleep(chunk_duration - processing_time)
            
    except KeyboardInterrupt:
        print("\nStopping stream...")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        audio_stream.close()

if __name__ == "__main__":
    main()
