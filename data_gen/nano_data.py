import os
import io
import time
import json
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

# Official Google GenAI SDK
from google import genai
from google.genai import types

# Custom imports from your environment
from system_hy import NARRATIVE_PROMPT_JSON, Iterative_T2I_PROMPT_QWEN
from utils import parse_llm_json
from nano_api import generate

GEMINI_APP_KEY =  
GEMINI_MODEL = "gemini-2.5-pro"

# Global save path
SAVE_FILE = "YOUR_PATH/nano"

# ==========================================
# 1. Concise Gemini Client (Official API)
# ==========================================
class GeminiAIClient:
    def __init__(self, api_key: str, max_tokens: int = 4096, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = genai.Client(api_key=api_key)
    
    def generate_content(self, contents: list) -> str:
        retry_times = 0
        while retry_times < 10:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.max_tokens,
                    )
                )
                if response.text is not None:
                    return response.text
            except Exception as e:
                print(f"Query failed, retry {retry_times + 1}, error: {e}")
                retry_times += 1
                time.sleep(3) 
        return "error"


# ==========================================
# 2. Global Configurations & Thread Local
# ==========================================
_thread_local = threading.local()


def _get_gemini_client():
    """Initialize a thread-local Gemini client to ensure thread safety."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = GeminiAIClient(
            api_key=GEMINI_APP_KEY, max_tokens=4096, model_name=GEMINI_MODEL
        )
    return _thread_local.client


def get_response_from_judge_model(prompt, first=False, image_file=None):
    """Constructs the contents list directly for the official Gemini API."""
    contents = [prompt]

    if image_file is not None:
        if not isinstance(image_file, list):
            image_file = [image_file]
        
        for img_path in image_file:
            # Read image directly as bytes
            with open(str(img_path), "rb") as f:
                img_bytes = f.read()
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            )

    response_text = _get_gemini_client().generate_content(contents)
    return response_text


def sanitize_for_path(text: str, max_len: int = 80) -> str:
    safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in text)
    safe = "_".join(safe.split())
    if not safe:
        safe = "sample"
    return safe[:max_len]


def is_sample_completed(example: dict, save_file: str = SAVE_FILE) -> bool:
    """A sample is considered complete if and only if output_subdir_name exists and contains execution_record.json."""
    user_input = example.get("user_input", "")
    example_id = example.get("id", "noid")
    output_subdir_name = f"{example_id}_{sanitize_for_path(user_input)}"
    subdir = os.path.join(save_file, output_subdir_name)
    record_path = os.path.join(subdir, "execution_record.json")
    return os.path.isdir(subdir) and os.path.isfile(record_path)


# ==========================================
# 3. Core Processing Logic
# ==========================================
def process_single_example(example: dict, file_name: str = "nano_omni"):
    user_input: str = example.get("user_input", "")
    user_input_image = example.get("user_input_image")
    example_id = example.get("id", "noid")

    output_subdir_name = f"{example_id}_{sanitize_for_path(user_input)}"

    flag = 0
    if user_input_image:
        first = True
        gen = False
    else:
        first = False
        gen = True

    step_count = 1
    step_groups = {}
    for attempt in range(5):
        try:
            text_input = NARRATIVE_PROMPT_JSON.replace('{text_input}', user_input)
            global_text_raw = get_response_from_judge_model(
                prompt=text_input,
                first=first,
                image_file=user_input_image,
            )
            global_text = parse_llm_json(global_text_raw)
            item = {
                "step": 1,
                'prompt': global_text['execution_plan'][step_count-1]['prompt'],
                'refine_prompt': "",
                'source_img': 'white.png',
                'target_img': '',
            }
            au = global_text['execution_plan'][0]['auxiliary_text']
            if au is not None and "None}, {'step_number'" in au:
                raise
            step_groups["1"] = item
            break
        except Exception:
            continue
            
    if len(step_groups) == 0:
        return

    save_file = SAVE_FILE
    successful_steps_record = []
    record_json_path = os.path.join(save_file, output_subdir_name, "execution_record.json")

    while True:
        if flag >= 30:
            break

        if flag % 2 == 0:
            temp = step_groups[str(step_count)]
            if gen:
                output_dir = os.path.join(save_file, output_subdir_name)
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(save_file, output_subdir_name, f"step{step_count}_{flag}.png")
  
                generate(
                    temp['prompt'] if temp.get("refine_prompt") == "" else temp["refine_prompt"],
                    out_path, 
                )
                step_groups[str(step_count)]['target_img'] = out_path

                flag += 1
                step_count += 1
            else:
                output_dir = os.path.join(save_file, output_subdir_name)
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(save_file, output_subdir_name, f"step{step_count}_{flag}.png")

                generate(
                    temp['prompt'] if temp.get("refine_prompt") == "" else temp["refine_prompt"], 
                    out_path, 
                    temp['source_img'], 
                )
              
                step_groups[str(step_count)]['target_img'] = out_path

                flag += 1
                step_count += 1
        else:
            step_temp = step_groups[str(step_count-1)]
            text_input = Iterative_T2I_PROMPT_QWEN.replace('{original_instruction}', global_text['execution_plan'][step_count-2]['prompt']).replace('{rewritten_prompt}', step_temp['refine_prompt'])
                        
            response_text_raw = get_response_from_judge_model(
                prompt=text_input,
                image_file=[step_temp['source_img'], step_temp['target_img']] 
            )
            response_text = parse_llm_json(response_text_raw)

            if not isinstance(response_text, dict) or response_text.get("previous_step_success") is None or response_text.get("refine_prompt") is None:
                step_count = step_count - 1
            else:
                step_groups[str(step_count-1)]['step'] = step_count-1
                step_groups[str(step_count-1)]['success'] = response_text["previous_step_success"]
                step_groups[str(step_count-1)]['reasoning'] = response_text["reasoning"]

                if not response_text["previous_step_success"]:
                    step_groups[str(step_count-1)]['refine_prompt'] = response_text['refine_prompt']
                    successful_steps_record.append((copy.deepcopy(step_groups[str(step_count-1)])))
                    
                    step_count = step_count - 1
                else:
                    step_groups[str(step_count-1)]['refine_prompt'] = ""
                    successful_steps_record.append((copy.deepcopy(step_groups[str(step_count-1)])))
                    if step_count == len(global_text["execution_plan"]) + 1:
                        break
                    else:
                        gen = False
                        try:
                            item = {
                                'prompt': global_text['execution_plan'][step_count-1]['prompt'],
                                'refine_prompt': "",
                                'source_img': step_temp['target_img'],
                                'target_img': '',
                            }
                        except Exception:
                            return
                        step_groups[str(step_count)] = item

            flag += 1

    os.makedirs(os.path.dirname(record_json_path), exist_ok=True)
    combined_data = {
        "user_input": user_input,
        "global_context": global_text,
        "steps": successful_steps_record  # Place the original list under the "steps" key
    }
    with open(record_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)


def load_dataset(jsonl_path: str):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data


def main():
    jsonl_path = "prompt40k.jsonl"
    dataset = load_dataset(jsonl_path)
    if not dataset:
        return

    # ================= Breakpoint resumption filtering logic =================
    original_len = len(dataset)
    # Filter out already completed samples
    dataset = [example for example in dataset if not is_sample_completed(example)]
    filtered_len = len(dataset)
    
    print(f"[*] Total samples: {original_len}")
    print(f"[*] Already completed: {original_len - filtered_len}")
    print(f"[*] Remaining to process: {filtered_len}")
    
    if not dataset:
        print("All tasks have been completed. Exiting.")
        return
    # =======================================================================

    num_workers = int(os.environ.get("WORKER_NUM", "8"))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_example, example, "nano_omni"): idx
            for idx, example in enumerate(dataset)
        }
        with tqdm(total=len(dataset), desc="prompts", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"[index {idx}] error: {e}")
                pbar.update(1)


if __name__ == "__main__":
    main()