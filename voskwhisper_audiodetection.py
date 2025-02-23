from voskwhisper import TranscriptionModel
import queue, signal, torch, logging, threading, pyaudio, time, numpy as np



logging.basicConfig(filename='voskwhisper.log', level=logging.INFO)



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



# Whisper model requires the conversion of int16 to float32 via audio_int16.astype(np.float32) / 32768.0

INT16_MAX_ABS_VALUE = 32768.0
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.064  # 64 milliseconds
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # 512 samples per chunk -> buffer size for silence detection.
ACTIVITY_BUFFER_SIZE = BLOCK_SIZE * 10 # 10 * 0.032  This buffer size is also used by vosk (~10000 or 20*0.032 = 0.64 seconds)

SILERO_SENSITIVITY = 0.8
MAX_SILERO_THREADS = 1  #   built-in thread count cap to avoid the program crashing without warning, unfortunately I was lied to, pytorch silero is not threadsafe, 
                        #   so we will force this value to be one.

class VoskWhisper:

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.audio_queue = queue.Queue()

        #initialize silero VAD:
        print("initializaing silero model")
        self.silero_vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            verbose=False,
            onnx=False
        )
        self.silero_sensitivity = SILERO_SENSITIVITY
        logging.info("Silero model intialized")

        # silence detection flags
        self.silero_working = False # flag set to true if silero is currently working
        self.is_silero_speech_active = False # Speech active from Silero

        """Silero silence detection thread pool"""
        self.silero_thread_pool = set()
        self.silero_max_thread_count = MAX_SILERO_THREADS

        """Thread management"""
        self.thread_lock = threading.Lock()
        self.thread_condition = threading.Condition(self.thread_lock)




        """ Transcription Model """
        self.transcription_model = None  # initialized later, right before transcription starts


    def start(self):
        """Method to start transcription process"""
        # start audio feed thread
        audio_thread = threading.Thread(target=self._audio_data_worker, args=())
        audio_thread.start()
        time.sleep(0.5)

        # Start transcription thread
        self._start_activity_monitor()



    def _audio_data_worker(self, target_sample_rate=16000, buffer_size=512):

        def preprocess_audio(chunk, original_sample_rate, target_sample_rate):
            """Preprocess audio chunk similar to feed_audio method."""
            if isinstance(chunk, np.ndarray):
                # Handle stereo to mono conversion if necessary
                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)
                # Resample to target_sample_rate if necessary
                if original_sample_rate != target_sample_rate:
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)
                # Ensure data type is int16
                chunk = chunk.astype(np.int16)
            else:
                # If chunk is bytes, convert to numpy array
                chunk = np.frombuffer(chunk, dtype=np.int16)
                # Resample if necessary
                if original_sample_rate != target_sample_rate:
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)
                    chunk = chunk.astype(np.int16)
            return chunk.tobytes()

        audio_interface = None
        stream = None
        device_sample_rate = 16000
        chunk_size = 1024  # 1024 bytes Increased chunk size for better performance
    
        audio_interface = pyaudio.PyAudio()
        default_device = audio_interface.get_default_input_device_info()  # uses default device

        stream = audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=device_sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
                input_device_index=default_device['index'],
            )

        print("audio initialized")
        buffer = bytearray()
        silero_buffer_size = 2 * buffer_size  # silero complains if too short, buffer size is 2x because int16 to bytes
        while True:
            data = stream.read(chunk_size)
            processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
            buffer += processed_data

            # Check if the buffer has reached or exceeded the silero_buffer_size (1024, 2048 bytes coming in, this is run 2x)
            while len(buffer) >= silero_buffer_size:
                to_process = buffer[:silero_buffer_size]
                buffer = buffer[silero_buffer_size:] 
                self.audio_queue.put(to_process)
                #print(to_process)

                """ at this point, the audio queue is filled with bytearrays of 1024 bytes representing 512 int16 frames """


    def _start_activity_monitor(self):
        """
        activity monitor loop, adds audio chunk from audio_queue to buffer of fixed size
        runs silence detection on each chunk
        activity detected -> transcription for the whole thread.
        """

        """Duped block, just to ensure flags are unset before running instance"""
        self.is_silero_speech_active = False # Speech active from Silero

        self.previous_block_active = False
        self.previous_block_active_2 = False


        activity_buffer = bytearray()
        activity_buffer_length = ACTIVITY_BUFFER_SIZE*2
        
        logging.info("Activity monitor initialized")



        """ Initialize transcription model """
        self.transcription_model = TranscriptionModel()


        while True:
            audio_chunk = self.audio_queue.get()

            if not self.is_silero_speech_active:
                with self.thread_condition:
                    """wait for a silero thread to finish when thread count is reached"""
                    while len(self.silero_thread_pool) >= self.silero_max_thread_count:
                        self.thread_condition.wait() 
                
                    data_copy = audio_chunk[:]
                    silero_thread = threading.Thread(
                        target=self._is_silero_speech,
                        args=(data_copy,))
                    
                    self.silero_thread_pool.add(silero_thread)
                    silero_thread.start()
                    #print("started silero thread")

            activity_buffer += audio_chunk

            if len(activity_buffer) >= activity_buffer_length:
                #print("running")
                if self.is_silero_speech_active:
                    # logic to transcribe block
                    #logging.info("Activity detected on this Block")
                    #print("Transcribing this block...")
                    """logic to transfer this block to a transcription queue"""

                    
                    self.transcription_model.append_latest_audio(activity_buffer[:])



                    # if self.previous_block_active:
                        # self.previous_block_active_2 = True
                    # else:
                        # self.previous_block_active_2 = False
                    self.previous_block_active = True
                    



                else:
                    #print("No activity on this block")
                    #logging.info("No activity detected on this Block")
                    # if previous block is an activity block, this block marks the end of the current transcription queue.

                    """logic to handle ending a transcription queue (running whisper and ending vosk)"""                    
                    # if self.previous_block_active_2 and not self.previous_block_active:
                        # self.transcription_model.run_whisper_on_latest_block()
                        # self.transcription_model.new_block()
                    

                    # if self.previous_block_active:
                        # self.previous_block_active_2 = True
                    # else:
                        # self.previous_block_active_2 = False
                    # self.previous_block_active = False

                    if self.previous_block_active:
                        self.transcription_model.run_whisper_on_latest_block()
                        self.transcription_model.new_block()
                    

                    

                # clears buffer and flags
                activity_buffer.clear()
                """update state of silero activity safely"""
                self._safe_silero_update(False)



    def _safe_silero_update(self, state):
        with self.thread_lock:
            # update
            self.is_silero_speech_active = state




    def _is_silero_speech(self, chunk):
        """
        STOLEN FROM REALTIMESTT, should be run in a separate thread whenever it is called.
        Returns true if speech is detected in the provided audio data
        1024 bytes / 2 bytes per sample = 512 samples, 512 samples / 16000 sample rate = 0.032 seconds per chunk 


        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """

        #print(f"starting silero {len(self.silero_thread_pool)}")

        try:
            audio_chunk = np.frombuffer(chunk, dtype=np.int16)
            audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
            vad_prob = self.silero_vad_model(
                torch.from_numpy(audio_chunk),
                SAMPLE_RATE).item()
            is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
            if is_silero_speech_active:
                
                """update state of silero activity safely"""
                self._safe_silero_update(True)
        except:
            print(bcolors.FAIL + "SILERO THREAD FAILED" + bcolors.ENDC)
            logging.fatal("SILERO THREAD FAILED")

        
        #print("ending silero")


        """Using condition to safely remove the thread from the pool"""
        #print("closing silero thread")
        try:
            self.thread_lock.acquire()
            self.silero_thread_pool.remove(threading.current_thread())
        finally:
            self.thread_lock.release()
        
        try:
            self.thread_condition.acquire()
            self.thread_condition.notify()
        finally:
            self.thread_condition.release()    

