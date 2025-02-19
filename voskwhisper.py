from vosk import Model, KaldiRecognizer
from faster_whisper import WhisperModel
#from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np, queue, os, sys, threading, json, time, logging
import queue
import tokenPredictorBlock, StorePredictionBlock
#from copy import deepcopy


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



INT16_MAX_ABS_VALUE = 32768.0

VOSK_MODEL_PATH ="/home/dave/live/vosk-model-en-us-0.22" # abs path required
WHISPER_MODEL= 'tiny'


OUTPUT_PATH = 'store.json'



MAX_VOSK_THREADS = 3
MAX_WHISPER_THREADS = 3




class TranscriptionModel:
    def __init__(self):
        self.queue = queue.Queue()
        self.latest_block = None

        self.jsonStore = StorePredictionBlock.StorePredictionBlock(OUTPUT_PATH)

            
        """Initialize Vosk Model"""
        vosk_model_path = VOSK_MODEL_PATH
        if not os.path.exists(vosk_model_path):  
            print("Vosk model path not found, see https://alphacephei.com/vosk/models")
            sys.exit(1)
        vosk_model = Model(vosk_model_path)  # Path to the Vosk model directory

        """Initialize Faster-Whisper Model"""
        whisper_model = WhisperModel(
            WHISPER_MODEL, 
            device="cuda", 
            compute_type="int8", 
            num_workers=MAX_WHISPER_THREADS
            )
        

        """initializes meta wave2vec2
        VoskWhisperBlock.wave2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", attn_implementation="flash_attention_2")
        VoskWhisperBlock.wave2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", attn_implementation="flash_attention_2")
        """
    

        """Initializes VoskWhisperBlock static models"""
        VoskWhisperBlock.vosk_model = vosk_model
        VoskWhisperBlock.whisper_model = whisper_model

        self.new_block()

    
        threading.Thread(target=self.__block_killer_daemon).start()





# modify to not kill, but store it using StorePredictionBlock

    def __block_killer_daemon(self):
        while True:
            try:
                timeout = 5
                waittime = 0
                block = self.queue.get()
                while not block.all_done:
                    time.sleep(0.1)

                    #handle timeout
                    waittime+=0.1
                    if waittime >= timeout:
                        break

                storeBlock = tokenPredictorBlock.tokenPredictorBlock(
                    vosk_t=block.vosk_times,
                    whisper_t=block.whisper_time,
                    vosk_str=block.vosk_strings,
                    whisper_str=block.whisper_string
                )
                print("storing")
                self.jsonStore.put(storeBlock)
                
                logging.info("block stored and removed from queue")
            except Exception as e:
                logging.fatal("ERROR IN TERMINATING BLOCK")
                print(bcolors.FAIL + "ERROR IN TERMINATING BLOCK" + bcolors.ENDC)





    def append_latest_audio(self, audiochunk):
        """appends audio data to the latest voskwhisperblock object in the queue"""
        self.latest_block.add(audiochunk)
        
    def new_block(self):
        """add a new block to queue and starts vosk on it"""
        self.latest_block = VoskWhisperBlock()
        self.queue.put(self.latest_block)
        self.latest_block._start_vosk()

    def run_whisper_on_latest_block(self):
        self.latest_block._start_whisper()



class VoskWhisperBlock:
    """
    Data structure class of a speechblock of VoskWhisper:
    - a list of strings with progressive vosk outputs
    - the final Whisper Model output string
    """

    vosk_model = None
    whisper_model = None
    wave2vec_model = None
    wave2vec_processor = None           # processor does not need to be copied by each thread created

    """Global thread pools"""
    max_threads = MAX_VOSK_THREADS

    vosk_pool = set()
    vosk_pool_lock = threading.Lock()
    vosk_pool_condition = threading.Condition(vosk_pool_lock)

    whisper_pool = set()
    whisper_pool_lock = threading.Lock()
    whisper_pool_condition = threading.Condition(whisper_pool_lock)



    def __init__(self):
        """Static variables for the vosk and whisper models must be set before initializing a new instance"""
        if VoskWhisperBlock.vosk_model == None or VoskWhisperBlock.whisper_model == None:
            raise Exception("Static models are uninitialized")
        else:
            self.Block = queue.Queue()
            self.WhisperBlock = bytearray()
        
            self.vosk_strings = []
            self.whisper_string = ""
            self.run_vosk = True    # flag denoting whether vosk should continue running on the current block, reset by whisper completion
            self.all_done = False   # flag used by block_killer_daemon to determine whether the current block can be discarded




            # system time keeping
            self.vosk_times = []
            self.whisper_time = None



    def add(self, chunk):
        self.Block.put(chunk)
        self.WhisperBlock += chunk
        
    def VoskStrings(self):
        return self.vosk_strings
        
    def WhisperString(self):
        return self.whisper_result
    
    def _start_vosk(self):
        """start vosk interpretation on the current block until whisper results finish"""
        """need global implementation on vosk thread pool"""

        #print(bcolors.OKBLUE + "requesting to start vosk thread" + bcolors.ENDC)
        #logging.info("requesting to start vosk thread")



        with VoskWhisperBlock.vosk_pool_condition:
            while len(VoskWhisperBlock.vosk_pool) >= VoskWhisperBlock.max_threads:
                VoskWhisperBlock.vosk_pool_condition.wait()

        vosk_thread = threading.Thread(
            target=self.vosk)
    

        vosk_thread.start()
        VoskWhisperBlock.vosk_pool.add(vosk_thread)

        #print(bcolors.OKGREEN + "started vosk thread" + bcolors.ENDC)
        logging.info("started vosk thread")



    def _start_whisper(self):
        """run whisper on the completed block"""
        """need global implementation on whisper thread pool"""
        self.run_vosk = False # this statement should be executed whenever whisper is executing the result

        #print(bcolors.OKBLUE + "requesting to start whisper thread" + bcolors.ENDC)
        logging.info("requesting to start whisper thread")

        with VoskWhisperBlock.whisper_pool_condition:
            while len(VoskWhisperBlock.whisper_pool) >= VoskWhisperBlock.max_threads:
                VoskWhisperBlock.whisper_pool_condition.wait()
            
        whisper_thread = threading.Thread(
            target=self.whisper)
    

        whisper_thread.start()
        VoskWhisperBlock.whisper_pool.add(whisper_thread)

        #print(bcolors.OKGREEN + "started whisper thread" + bcolors.ENDC)
        logging.info("started whisper thread")





    def vosk(self):
        #print("running vosk")
        try:
            recognizer = KaldiRecognizer(VoskWhisperBlock.vosk_model, 16000)  # individual kaldirecognizers per thread - problem with initialization time
            
            while self.run_vosk:

                try:
                    data = self.Block.get(timeout=0.1)
                    audio_16 = np.frombuffer(data, dtype=np.int16)
                
                    recognizer.AcceptWaveform(bytes(audio_16))
                    partial_result = json.loads(recognizer.PartialResult())
                    partial_text = partial_result["partial"]

                    self.vosk_strings.append(partial_text)
                    t = time.perf_counter()
                    self.vosk_times.append(t)

                    print("PARTIAL: " + partial_text)


                except queue.Empty:
                    if not self.run_vosk:
                        break
                
            #print(bcolors.WARNING + "ended vosk thread" + bcolors.ENDC)
            logging.info("ended vosk thread")

            # handle thread removal from static pool
            with VoskWhisperBlock.vosk_pool_condition:
                VoskWhisperBlock.vosk_pool.remove(threading.current_thread())
                VoskWhisperBlock.vosk_pool_condition.notify()
        
        except Exception as e:
            print(e)
    


    def whisper(self):
        try:
            audio_16 = np.frombuffer(self.WhisperBlock, dtype=np.int16)
            audio_32 = audio_16.astype(np.float32) / INT16_MAX_ABS_VALUE

            segments, _ = VoskWhisperBlock.whisper_model.transcribe(audio_32, language="en", beam_size=1, word_timestamps=True)
            
            #print(bcolors.WARNING + "whisper model finished" + bcolors.ENDC)
            #logging.info("whisper model finished")

            self.whisper_time = time.perf_counter()

            for segment in segments:
                self.whisper_string += segment.text
                print(segment.text)

            #print(bcolors.WARNING + "ended whisper thread" + bcolors.ENDC)
            logging.info("ended whisper thread")


            with VoskWhisperBlock.whisper_pool_condition:
                VoskWhisperBlock.whisper_pool.remove(threading.current_thread())
                VoskWhisperBlock.whisper_pool_condition.notify()

            self.all_done = True

        except Exception as e:
            print(e)
