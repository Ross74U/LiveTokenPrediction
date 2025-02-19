# has components: 
# json Writer - easiler to parse and process later, DB kinda overkill i think

import tokenPredictorBlock
import json, os, datetime

class StorePredictionBlock:
    
    def __init__(self, filename):
        self.filename = filename
        self.count = 0

    def put(self, block):
        blockdict = {
            "voskstrs": [],
            "whisperstr": "",
            "voskt": 0,
            "whispert": 0
        
    # hey this is a comment written in neovim with nothing else, lol. okay nvm it is actually written in this super duper awesome nvim prerice NVChad.

        blockdict["voskstrs"] = block.vosk_strings
        blockdict["whisperstr"] = block.whisper_string
        blockdict["voskt"] = block.vosk_times
        blockdict["whispert"] = block.whisper_time

        name = f"block{self.count}"
        
        # Create a single entry with name as key
        entry = {name: blockdict}
        
        # Check file size (100MB limit as example)
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes
        current_file = 'results.jsonl'
        
        if os.path.exists(current_file) and os.path.getsize(current_file) >= MAX_FILE_SIZE:
            # Create new filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_file = f'results_{timestamp}.jsonl'
            os.rename(current_file, new_file)
        
        # Append the new entry as a single line
        with open(current_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')  # Add newline between entries
        
        self.count += 1

    def read_specific_block(filename, block_name):
        """Utility function to read a specific block without loading entire file"""
        with open(filename, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if block_name in entry:
                    return entry[block_name]
        return None
