import json
import re
import json_repair

def parse_llm_json(response_text):
    try:
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        json_text = answer_match.group(1).strip() if answer_match else response_text
        json_text = json_repair.loads(json_text)
        return json_text
        # return json_repair.loads(response_text)
    except:
        return None

def parse_score_json(response_text):
    try:
        answer_match = re.search(r"<score>(.*?)</score>", response_text, re.DOTALL)
        json_text = answer_match.group(1).strip() if answer_match else None
        json_text = json_repair.loads(json_text)
        return json_text
        # return json_repair.loads(response_text)
    except:
        return None

def parse_str_json(response_text):
    try:
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        json_text = answer_match.group(1).strip() if answer_match else None
        return json_text
    except:
        return None


