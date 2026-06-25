import re
from typing import Union

delete_number = re.compile(r'[0-9]*\s*\.\s*')
delete_space = re.compile(r'\s+')

def wplc_score(
        references: Union[str, list[list[str]]],
        predictions: Union[str, list[list[str]]],
        **kwargs
        ) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions, str):
        predictions = [[predictions]]
    # print(predictions, "pred")
    # print(references, "refer")
    total = 0
    correct = 0
    for preds, target in zip(predictions, references):
        if preds[0] == target:
            correct += 1
        total += 1
    return correct/total


class WPLCTextfilter:
    
    def __init__(self):
        return 
    
    def apply(self, resps: Union[str, list[list[str]]], docs, **kwargs):
        filtered_resps = []
        for resp_topk in resps:
            filtered_topk = []
            for resp in resp_topk:
                resp = delete_number.sub("", resp)
                resp = delete_space.sub(" ", resp)
                resp = resp.split(" ")[0].strip()
                filtered_topk.append(resp)
            filtered_resps.append(filtered_topk)
        return filtered_resps