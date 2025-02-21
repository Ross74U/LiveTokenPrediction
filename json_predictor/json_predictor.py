# this program reads the results.jsonl and does token prediction along with other things to append to a new jsonl
"""
contents of results.jsonl
each line is a json file
{"block3": {"voskstrs": [], "whisperstr": "", "voskt": [], "whispert": null}}
"""
import json
from prediction.predictor_base import Predictor, TokenInfo
from prediction.gpt2_transformer import gpt2TokenPredictor
from typing import TypedDict, Any, get_type_hints, TextIO
import time


class BlockInfo(TypedDict):
    voskstr: list[str]
    whisperstr: str
    voskt: list[float]
    whispert: float
    token_prediction_times: list[float]
    tokens = list[TokenInfo]

def ProcessJson(predictor: Predictor, filepath: str, output_path: str):

    def _jsonl_write(output: BlockInfo, o: TextIO):
        json.dump(output, o)
        o.write('\n')

    def _is_typed_dict_compatible(typed_dict_cls: type, data: dict) -> bool:
        """
        generic:
        Check if a dictionary is compatible with a given TypedDict.
        """
        annotations = get_type_hints(typed_dict_cls)
        for key, expected_type in annotations.items():
            if key not in data:
                return False  # Missing required key
            if data[key] is None or data == "" or value == [] or value == {}:
                continue
            if not isinstance(data[key], expected_type):
                return False  # Value type mismatch
        return True      


    f = open(filepath, "r")
    o = open(output_path, "a")

    for block in f:
        """Preprocess data so it fits the typehints of BlockInfo alias""" 
        data: dict = next(iter(json.loads(block.strip()).values())) # Preprocess data so it fits the typehints of BlockInfo alias
        data["token_prediction_times"] = []
        data["tokens"] = []

        # if not _is_typed_dict_compatible(TokenInfo, data):
        #     raise ValueError("Dictionary not compatible with BlockInfo!")
        #     return

        current_block: BlockInfo = data
        
        for string in current_block["voskstrs"]:
            if string == "":
                continue
            t1: float = time.perf_counter()
            info: TokenInfo = predictor.next_token(string)
            t2: float = time.perf_counter()
            current_block["tokens"].append(info)
            current_block["token_prediction_times"].append(t2-t1)
        
        _jsonl_write(current_block, o)

    f.close()
    o.close()





if __name__ == "__main__":
    filepath = "results.jsonl"
    output_path = "results_out.jsonl"
    predictor: Predictor = gpt2TokenPredictor()
    
    ProcessJson(predictor, filepath, output_path)

