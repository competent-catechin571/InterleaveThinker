#!/usr/bin/env python3
"""
Generate iterative text and image outputs using Qwen Image Edit and Gemini Judge.
Loads prompts from HuggingFace dataset (primerL/UEval-all).
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tempfile
import torch
from PIL import Image
from system import NARRATIVE_PROMPT_JSON, Iterative_T2I_PROMPT_QWEN
from utils import parse_llm_json
from diffusers import Flux2KleinPipeline
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen3_vl_api import predict as qwen3_vl_predict

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: 'datasets' library not found. Install with: pip install datasets")
    load_dataset = None


class IterativeImageGenerator:
    """Generator for iterative image editing and judging using Qwen and Gemini."""

    def __init__(self, model_name: str, planner_path, critic_path, klein_ckpt_path: str, device: str):
        self.device = device

        self.planner = Qwen3VLForConditionalGeneration.from_pretrained(
            planner_path, 
            dtype="auto", device_map=self.device)
        
        self.critic = Qwen3VLForConditionalGeneration.from_pretrained(
            critic_path, 
            dtype="auto", device_map=self.device)
        
        self.qwen_processor = AutoProcessor.from_pretrained(critic_path)

        print(f"[{self.device}] Loading Kelvin Pipeline from {klein_ckpt_path}...")
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            klein_ckpt_path, 
            torch_dtype=torch.bfloat16
        )

        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        print(f"[{self.device}] Pipeline loaded successfully.")

    def construct_msgs(self, text: str, imgs: Optional[List[str]]) -> List[Dict]:
        user_content = []
        if imgs is not None:
            for img in imgs:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": text})

        messages = [
            {"role": "user", "content": user_content}
        ]
        return messages

    def get_response_from_judge_model_planner(self, prompt: str, image_file: Optional[str] = None) -> str:
        image_inputs = []
        temp_white_image_path = None  
        try:
            if image_file is not None:
                if not isinstance(image_file, list):
                    image_file = [image_file]
                    
                if len(image_file) == 1:
                    img_path = str(image_file[0])
                    with Image.open(img_path) as img:
                        width, height = img.size
                    white_img = Image.new('RGB', (width, height), color='white')
                    
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    temp_white_image_path = tmp_file.name
                    tmp_file.close() 
                    
                    white_img.save(temp_white_image_path, format="PNG")
                    image_inputs.append(temp_white_image_path)
                    
                for img in image_file:
                    image_inputs.append(str(img))
            else:
                image_inputs = None
                
            messages = self.construct_msgs(prompt, image_inputs)
            response_text = qwen3_vl_predict(self.planner, self.qwen_processor, messages)  # type: ignore 
            return response_text
            
        finally:
            if temp_white_image_path and os.path.exists(temp_white_image_path):
                os.remove(temp_white_image_path)
    
    def get_response_from_judge_model_critic(self, prompt: str, image_file: Optional[str] = None) -> str:
        image_inputs = []
        temp_white_image_path = None  
        try:
            if image_file is not None:
                if not isinstance(image_file, list):
                    image_file = [image_file]
                    
                if len(image_file) == 1:
                    img_path = str(image_file[0])
                    with Image.open(img_path) as img:
                        width, height = img.size
                    white_img = Image.new('RGB', (width, height), color='white')
                    
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    temp_white_image_path = tmp_file.name
                    tmp_file.close() 
                    
                    white_img.save(temp_white_image_path, format="PNG")
                    image_inputs.append(temp_white_image_path)
                    
                for img in image_file:
                    image_inputs.append(str(img))
            else:
                image_inputs = None
                
            messages = self.construct_msgs(prompt, image_inputs)
            response_text = qwen3_vl_predict(self.critic, self.qwen_processor, messages)  # type: ignore 
            return response_text
            
        finally:
            if temp_white_image_path and os.path.exists(temp_white_image_path):
                os.remove(temp_white_image_path)

    def generate(
        self,
        prompt_item: Dict[str, Any],
        output_image_dir: str,
        retry_delay: float = 3.0,
        max_attempts: int = 3,
        max_step_iterations: int = 3,  
    ) -> Tuple[List[str], str]:
        """
        Execute the iterative generation and judging process for a single prompt.
        """
        user_input = prompt_item["prompt"]
        user_input=user_input.replace(' both visually and textually', '')
        prompt_id = prompt_item["id"]
        attempt = 0

        current_out_dir = os.path.join(output_image_dir, str(prompt_id))
        os.makedirs(current_out_dir, exist_ok=True)

        global_text = None
        for attempt in range(max_attempts):
            try:
                text_input = NARRATIVE_PROMPT_JSON.replace('{text_input}', user_input)
                global_text_raw = self.get_response_from_judge_model_planner(prompt=text_input, image_file=None)
                global_text = parse_llm_json(global_text_raw)
                for item in global_text['execution_plan']:
                    pp = item['prompt']
                    au = item['auxiliary_text']
                    ii = item['instruction']
                    if au is not None and "None}, {'step_number'" in au:
                        raise
                break
                
            except Exception as e:
                print(f"[{self.device}] Attempt {attempt} failed with error: {e}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                # continue
        if not isinstance(global_text, dict):
            raise RuntimeError(f"[{self.device}] Failed to get Planner execution plan after {max_attempts} attempts for ID {prompt_id}")

        output_text = ""
        step_groups = {}
        
        current_source_img = 'white.png'
        execution = global_text['execution_plan']

        for i, plan_item in enumerate(execution):
            step_count = i + 1
            original_prompt = plan_item['prompt']
            
            step_groups[str(step_count)] = {
                'prompt': original_prompt,
                'refine_prompt': "",
                'source_img': current_source_img,
                'target_img': '',
            }
            
            current_refine_prompt = ""
            target_img_path = ""

            for attempt in range(max_step_iterations):
                out_path = os.path.join(current_out_dir, f'step{step_count}.png')
                current_seed = step_count * 100 + attempt 
                
                prompt_temp = current_refine_prompt if current_refine_prompt != "" else original_prompt

                if i == 0:  
                    image = self.pipeline(
                        prompt=prompt_temp,
                        height=1024,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=torch.Generator(device=self.device).manual_seed(current_seed)
                    ).images[0]
                else: 
                    image1 = Image.open(current_source_img).convert("RGB")
                    image = self.pipeline(
                        image=image1,
                        prompt=prompt_temp,
                        height=1024,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=torch.Generator(device=self.device).manual_seed(current_seed)
                    ).images[0]

                image.save(out_path)
                target_img_path = out_path
                step_groups[str(step_count)]['target_img'] = target_img_path

                if attempt == max_step_iterations - 1:
                    print(f"[{self.device}] Step {step_count} reached max iterations ({max_step_iterations}). Forcing pass.")
                    break

                # Critic
                text_input = Iterative_T2I_PROMPT_QWEN.replace(
                    '{original_instruction}', original_prompt
                ).replace('{rewritten_prompt}', prompt_temp)
                
                response_text = None
                for api_attempt in range(max_attempts):
                    try:
                        response_text_raw = self.get_response_from_judge_model_critic(
                            prompt=text_input, 
                            image_file=[current_source_img, target_img_path] 
                        )
     
                        response_text = parse_llm_json(response_text_raw)
                        refine_prompt = response_text['refine_prompt']
                        judge = response_text['previous_step_success']
                        if not isinstance(refine_prompt, str) or not isinstance(judge, bool):
                            raise ValueError("Invalid refine prompt")
                        break
                    except Exception as e:
                        print(f"[{self.device}] Attempt {api_attempt} failed with error: {e}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                
                if not isinstance(judge, bool) or not isinstance(refine_prompt, str):
                    pass 
                if judge:
                    break
                else:
                    current_refine_prompt = refine_prompt
                    step_groups[str(step_count)]['refine_prompt'] = current_refine_prompt

            current_source_img = target_img_path
                   
        record_json_path = f'{current_out_dir}/execution_record.json'
        with open(record_json_path, 'w', encoding='utf-8') as f:
            json.dump(step_groups, f, indent=4, ensure_ascii=False)
        
        for i in range(len(global_text['execution_plan'])):
            if global_text['execution_plan'][i]['auxiliary_text'] and global_text['execution_plan'][i]['auxiliary_text'] != 'None':
                output_text+=global_text['execution_plan'][i]['auxiliary_text']
            else:
                output_text+=global_text['execution_plan'][i]['instruction']
        out_paths = []
        for it in step_groups.keys():
            out_paths.append(step_groups[it]['target_img'])
        return out_paths, output_text

def load_dataset_from_hf(
    hf_dataset_id: str,
    domains: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("The 'datasets' library is required. Install with: pip install datasets")

    try:
        dataset = load_dataset(hf_dataset_id, split="test")
        data = [dict(item) for item in dataset]

        if domains:
            domains_set = set(d.lower() for d in domains)
            data = [item for item in data if item.get("task", "").lower() in domains_set]

        if limit:
            data = data[:limit]

        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from HuggingFace: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate iterative text and images using Qwen and Gemini Judge")
    parser.add_argument("--hf_dataset", default="primerL/UEval-all", help="HuggingFace dataset ID")
    parser.add_argument("--domains", nargs="+", default=None, help="Filter by task types")
    parser.add_argument("--output_path", required=True, help="Path to save output JSON file")
    parser.add_argument("--output_image_dir", required=True, help="Directory to save generated images")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name for Judge")
    parser.add_argument("--planner_path", default=None, help="Path to Planner model")
    parser.add_argument("--critic_path", default=None, help="Path to Critic model")
    parser.add_argument("--klein_ckpt", default="black-forest-labs/FLUX.2-klein-9B", help="Path to FLUX.2-klein model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process")
    parser.add_argument("--retry_delay", type=float, default=3.0, help="Seconds to wait between retry attempts")
    parser.add_argument("--max_attempts", type=int, default=5, help="Maximum retry attempts per prompt")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save checkpoint every N items")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    print(f"[{device}] Init: Local Rank {local_rank}, World Size {world_size}")

    if local_rank == 0:
        print(f"\n{'='*60}\nLoading dataset...\n{'='*60}")
        
    data = load_dataset_from_hf(hf_dataset_id='zlab-princeton/UEval', domains=args.domains, limit=args.limit)

    if not data:
        print(f"[{device}] No data to process. Exiting.")
        return

    data = sorted(data, key=lambda x: str(x.get("id", "")))
    data = data[local_rank::world_size]

    print(f"[{device}] Will process {len(data)} items (Total GPUs: {world_size})")

    generator = IterativeImageGenerator(
        model_name=args.model,
        planner_path=args.planner_path,
        critic_path=args.critic_path,
        klein_ckpt_path=args.klein_ckpt,
        device=device
    )

    base_output_path = Path(args.output_path)
    output_path = base_output_path.parent / f"{base_output_path.stem}_rank{local_rank}{base_output_path.suffix}"

    outputs = []
    global_processed_ids = set()

    search_pattern = f"{base_output_path.stem}_rank*{base_output_path.suffix}"
    existing_files = list(base_output_path.parent.glob(search_pattern))
    
    if existing_files:
        for file_path in existing_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                
                for item in file_data:
                    global_processed_ids.add(str(item["id"]))
                
                if file_path.name == output_path.name:
                    outputs = file_data
                    print(f"[{device}] Loaded {len(outputs)} items from OWN file: {file_path.name}")
                    
            except Exception as e:
                print(f"[{device}] ⚠️ Failed to parse {file_path.name}: {e}")

        print(f"[{device}] Global processed IDs across all ranks: {len(global_processed_ids)}")
    # =========================================================================

    for idx, item in enumerate(data):
        item_id = str(item.get("id"))

        if item_id in global_processed_ids:
            print(f"[{device}] ⏭️ Skipping globally processed ID {item_id}")
            continue

        print(f"[{device}] [{idx + 1}/{len(data)}] Processing ID {item_id}")

        try:
            image_paths, text_response = generator.generate(
                prompt_item=item,
                output_image_dir=args.output_image_dir,
                retry_delay=args.retry_delay,
                max_attempts=args.max_attempts,
            )

            output_item = {
                "id": item_id,
                "prompt": item.get("prompt", ""),
                "task": item.get("task", ""),
                "image_answer": image_paths,
                "text_answer": text_response,
            }

            outputs.append(output_item)
            global_processed_ids.add(item_id)

            if args.checkpoint_interval > 0 and (len(outputs) % args.checkpoint_interval == 0):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_output_path = output_path.with_suffix('.tmp')
                with open(tmp_output_path, "w", encoding="utf-8") as f:
                    json.dump(outputs, f, ensure_ascii=False, indent=2)
                os.replace(tmp_output_path, output_path)

        except Exception as e:
            print(f"\n[{device}] ❌ Error processing ID {item_id}: {e}")
            continue 

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix('.tmp')
    with open(tmp_output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    os.replace(tmp_output_path, output_path)
    
    print(f"[{device}] ✅ Task finished. Saved final output to: {output_path}")

if __name__ == "__main__":
    main()