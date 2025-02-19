# this program reads the results.jsonl and does token prediction along with other things to append to a new jsonl
"""
contents of results.jsonl
each line is a json file
{"block3": {"voskstrs": [], "whisperstr": "", "voskt": [], "whispert": null}}
"""
import json
from prediction.gpt2_transformer import gpt2TokenPredictor

filepath = "results.jsonl"
blocks = [] # stores all the output blocks of the json_predictor program, to be written out to a .jsonl file

with open(filepath, "r") as f:
    for block in f:
        data = json.loads(block.strip())
        block = data[list(data.keys())[0]]
        voskstrs = block["voskstrs"]  
        whisperstr = block["whisperstr"]
        vosk_time = block["voskt"]
        whisper_time = block["whispert"]
        token_prediction_time = []  # system timestamp of prediction from each string in voskstr
