# holds system timing information
# every single vosk partial + it's completion time
# every single

class tokenPredictorBlock:
    def __init__(self):
        self.vosk_times = []
        self.whisper_time = None
        
        self.vosk_strings = []
        self.whisper_string = None


    def __init__(self, vosk_t, whisper_t, vosk_str, whisper_str):
        if len(vosk_str) != len(vosk_t):
            raise Exception("Vosk strs len != Vosk times!\n aka u done fukd up")
        
        self.vosk_times = vosk_t
        self.whisper_time = whisper_t
        
        self.vosk_strings = vosk_str
        self.whisper_string = whisper_str
