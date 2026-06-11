import os
import io
import time
import json
import copy
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

# Official Google GenAI SDK
from google import genai
from google.genai import types

# Custom imports from your environment
from system_hy import NARRATIVE_PROMPT_JSON, Iterative_T2I_PROMPT_QWEN
from utils import parse_llm_json
from diffusers import Flux2KleinPipeline

SAVE_FILE = "YOUR_PATH/klein"
API_KEY = 

# ==========================================
# 1. Concise Gemini Client (Official API)
# ==========================================
class GeminiAIClient:
    def __init__(self, api_key: str, max_tokens: int = 4096, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = genai.Client(api_key=api_key)
        print(f"Initialized official Gemini Client, model: {self.model_name}")
    
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
# 2. Utility Functions
# ==========================================
def gpu_burner_worker(local_rank, matrix_size=8192, sleep_time=0.01):
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Create large FP32 matrices to force high-intensity GPU computation
    a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
    b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
    
    while True:
        try:
            for _ in range(6):
                c = torch.matmul(a, b)
            torch.cuda.synchronize(device)
            if sleep_time > 0:
                time.sleep(sleep_time)
        except Exception:
            pass


def get_completed_samples(save_file: str) -> set:
    """Scan the directory once to relieve I/O pressure and achieve fast breakpoint resumption"""
    completed = set()
    if not os.path.exists(save_file):
        return completed
    try:
        for subdir in os.listdir(save_file):
            record_path = os.path.join(save_file, subdir, "execution_record.json")
            if os.path.isfile(record_path):
                completed.add(subdir)
    except Exception:
        pass
    return completed


def sanitize_for_path(text: str, max_len: int = 80) -> str:
    text = text.replace(" both visually and textually", "")
    safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in text)
    safe = "_".join(safe.split())
    if not safe:
        safe = "sample"
    return safe[:max_len]


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


def get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return world_size, rank, local_rank


# ==========================================
# 3. Main Generator Class
# ==========================================
class KelvinOmniDataGenerator:
    """Hold independent client and pipeline per GPU/process."""

    def __init__(self, api_key: str, model_name: str, pipeline_ckpt_path: str, device: str):
        self.device = device
        self.client = GeminiAIClient(api_key=api_key, max_tokens=4096, model_name=model_name)
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            pipeline_ckpt_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def get_response_from_judge_model(self, prompt, first=False, image_file=None):
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

        response_text = self.client.generate_content(contents)
        return response_text

    def process_single_example(self, example: dict, file_name: str = "kelvin_omni"):
        user_input: str = example.get("user_input", "")
        user_input_image = example.get("user_input_image", None)
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
                global_text_raw = self.get_response_from_judge_model(
                    prompt=text_input,
                    first=first,
                    image_file=None,
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
                current_seed = flag
                if gen:
                    out_path = os.path.join(save_file, output_subdir_name, f"step{step_count}_{flag}.png")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    image = self.pipeline(
                        prompt=temp['prompt'] if temp.get("refine_prompt") == "" else temp["refine_prompt"],
                        height=1024,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=torch.Generator(device=self.device).manual_seed(current_seed),
                    ).images[0]
                    image.save(out_path)

                    step_groups[str(step_count)]['target_img'] = out_path
                    flag += 1
                    step_count += 1
                else:
                    out_path = os.path.join(save_file, output_subdir_name, f"step{step_count}_{flag}.png")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    image1 = Image.open(temp['source_img']).convert("RGB")
                    image = self.pipeline(
                        image=image1,
                        prompt=temp['prompt'] if temp.get("refine_prompt") == "" else temp["refine_prompt"],
                        height=1024,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=torch.Generator(device=self.device).manual_seed(current_seed),
                    ).images[0]

                    image.save(out_path)
                    step_groups[str(step_count)]['target_img'] = out_path
                    flag += 1
                    step_count += 1
            else:
                step_temp = step_groups[str(step_count-1)]
                text_input = Iterative_T2I_PROMPT_QWEN.replace('{original_instruction}', global_text['execution_plan'][step_count-2]['prompt']).replace('{rewritten_prompt}', step_temp['refine_prompt'])
                            
                response_text_raw = self.get_response_from_judge_model(
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
            "steps": successful_steps_record
        }
        with open(record_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)


# ==========================================
# 4. Main Execution
# ==========================================
def main():
    jsonl_path = "prompt40k.jsonl"
    pipeline_ckpt_path = "black-forest-labs/FLUX.2-klein-9B"
    dataset = load_dataset(jsonl_path)
    if not dataset:
        return

    world_size, rank, local_rank = get_dist_info()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Initialize Generator first to load model to GPU
    generator = KelvinOmniDataGenerator(
        api_key=API_KEY, 
        model_name="gemini-2.5-pro",
        pipeline_ckpt_path=pipeline_ckpt_path,
        device=device,
    )

    if torch.cuda.is_available():
        ctx = mp.get_context('spawn')
        burner_process = ctx.Process(
            target=gpu_burner_worker, 
            args=(local_rank, 8192, 0.01), 
            daemon=True 
        )
        burner_process.start()

    completed_set = get_completed_samples(SAVE_FILE)

    pending_dataset = []
    for idx, example in enumerate(dataset):
        if idx % world_size != rank:
            continue
            
        user_input = example.get("user_input", "")
        example_id = example.get("id", "noid")
        output_subdir_name = f"{example_id}_{sanitize_for_path(user_input)}"
        
        if output_subdir_name not in completed_set:
            pending_dataset.append(example)

    with tqdm(total=len(pending_dataset), desc=f"Rank {rank} pending prompts", dynamic_ncols=True) as pbar:
        for example in pending_dataset:
            generator.process_single_example(example)
            pbar.update(1)


if __name__ == "__main__":
    main()