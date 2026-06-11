#!/usr/bin/env python3
import os
import time
import tempfile
import torch
import json
import re
import argparse
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from diffusers import Flux2KleinPipeline
from UEval.system import NARRATIVE_PROMPT_JSON, Iterative_T2I_PROMPT_QWEN, GUIDANCE_GLOBAL_PROMPT_JSON
from UEval.utils import parse_llm_json
from UEval.qwen3_vl_api import predict as qwen3_vl_predict

class SingleSampleGenerator:
    """Generator for iterative image editing and judging using Qwen and Nano."""

    def __init__(self, planner_ckpt: str, critic_ckpt: str, qwen_processor_ckpt: str, device: str = "cuda:0"):
        self.device = device
        print(f"[{self.device}] Loading Models...")

        self.planner = Qwen3VLForConditionalGeneration.from_pretrained(
            planner_ckpt, dtype="auto", device_map=self.device, attn_implementation="flash_attention_2"
        )
        
        self.critic = Qwen3VLForConditionalGeneration.from_pretrained(
            critic_ckpt, dtype="auto", device_map=self.device, attn_implementation="flash_attention_2"
        )
        
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_processor_ckpt)

        print(f"[{self.device}] All models loaded successfully.")

        print(f"[{self.device}] Loading Kelvin Pipeline")
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B", 
            torch_dtype=torch.bfloat16
        )
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        print(f"[{self.device}] Pipeline loaded successfully.")

    def construct_msgs(self, text: str, imgs: Optional[List[str]]) -> List[Dict]:
        user_content: List[Dict[str, Any]] = []
        if not imgs:
            user_content.append({"type": "text", "text": text})
            return [{"role": "user", "content": user_content}]

        num_imgs = len(imgs)

        parts = re.split(r'(?i)<image>', text)
        for i, part in enumerate(parts):
            if part:
                user_content.append({"type": "text", "text": part})
            
            if i < len(imgs):
                if num_imgs >= 5:
                    user_content.append({
                        "type": "image", 
                        "image": imgs[i],
                        "max_pixels": 384 * 384
                    })
                else:
                    user_content.append({"type": "image", "image": imgs[i]})
        return [{"role": "user", "content": user_content}]

    def _get_qwen_response(self, model, prompt: str, image_file: Optional[List[str]] = None) -> str:
        image_inputs = []

        if image_file is not None:
            if isinstance(image_file, str):
                image_file = [image_file]
                
            for img in image_file:
                image_inputs.append(str(img))
        else:
            image_inputs = None
            
        messages = self.construct_msgs(prompt, image_inputs)
        response_text = qwen3_vl_predict(model, self.qwen_processor, messages)
        return response_text

    def generate_sequence(
        self,
        user_prompt: str,
        output_dir: str,
        input_image = None,
        max_step_iterations: int = 3,  
        retry_delay: float = 3.0,
        max_attempts: int = 100,
    ) -> List[Dict[str, str]]:
        os.makedirs(output_dir, exist_ok=True)
        user_prompt = user_prompt.replace(' both visually and textually', '')
        initial_white_img = "data_gen/white.png"
        if input_image is None:
            planner_prompt = NARRATIVE_PROMPT_JSON
            current_source_img = [initial_white_img]  
        else:
            planner_prompt = GUIDANCE_GLOBAL_PROMPT_JSON
            current_source_img = input_image  
        
        # Planning
        global_text = None
        for attempt in range(max_attempts):
            try:
                text_input = planner_prompt.replace('{text_input}', user_prompt)
                global_text_raw = self._get_qwen_response(self.planner, prompt=text_input, image_file=input_image)
                global_text = parse_llm_json(global_text_raw)
                for ite in global_text['execution_plan']:
                    pp = ite['step_number']
                    pp = ite['instruction']
                    pp = ite['prompt']
                    pp = ite['auxiliary_text']
                    if pp is not None and "None}, {'step_number'" in pp:
                        raise ValueError("Invalid execution plan format")
                break
            except Exception as e:
                print(f"[{self.device}] Planner attempt {attempt} failed: {e}")
                time.sleep(retry_delay)

        if not isinstance(global_text, dict):
            raise RuntimeError("Failed to get Planner execution plan.")
        
        print('Finish planning')

        interleaved_sequence = []

        for i, plan_item in enumerate(global_text['execution_plan']):
            step_num = plan_item['step_number']
            step_count = step_num
            original_prompt = plan_item['prompt'] if plan_item['prompt'] is not None else plan_item['auxiliary_text']
            current_refine_prompt = original_prompt
            target_img = ""

            for attempt in range(max_step_iterations):
                target_img = f'{output_dir}/step{step_count}_{attempt + 1}.png'
                if step_count == 1 and input_image is None:
                    image = self.pipeline(
                        prompt=current_refine_prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=torch.Generator(device=self.device).manual_seed(int(i*1000))
                    ).images[0]
                else:  
                    image1 = Image.open(current_source_img[-1]).convert("RGB")
                    image = self.pipeline(
                        image=image1,
                        prompt=current_refine_prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=torch.Generator(device=self.device).manual_seed(int(i*1000))
                    ).images[0]
                image.save(target_img)
                    
                # Critic
                text_input = Iterative_T2I_PROMPT_QWEN.replace(
                    '{original_instruction}', original_prompt
                ).replace('{rewritten_prompt}', current_refine_prompt)
                text_input = f'<image><image>\n{text_input}'
                
                judge = None
                refine_prompt = current_refine_prompt
                
                for api_attempt in range(max_attempts):
                    try:
                        print([current_source_img[-1], target_img])
                        response_text_raw = self._get_qwen_response(
                            self.critic, 
                            prompt=text_input, 
                            image_file=[current_source_img[-1], target_img]
                        )
                        response_text = parse_llm_json(response_text_raw)
                        refine_prompt = response_text['refine_prompt']
                        judge = response_text['previous_step_success']
                        if not isinstance(refine_prompt, str) or not isinstance(judge, bool):
                            raise ValueError("Invalid refine prompt")
                        break
                    except Exception as e:
                        print(f"[{self.device}] Critic attempt {api_attempt} failed: {e}")
                        time.sleep(retry_delay)

                if not isinstance(judge, bool) or not isinstance(refine_prompt, str):
                    pass

                if judge:
                    break
                else:
                    current_refine_prompt = refine_prompt

            current_source_img.append(target_img)
            
            text_content = plan_item['auxiliary_text'] if plan_item.get('auxiliary_text') and plan_item['auxiliary_text'] != 'None' else plan_item['instruction']
            interleaved_sequence.append({
                "step": step_count,
                "text": text_content,
                "image": target_img
            })

        return interleaved_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SingleSampleGenerator")
    
    parser.add_argument(
        "--planner_ckpt", 
        type=str, 
        default="ckpt/planner_sft",
        help="Path to the planner checkpoint"
    )
    parser.add_argument(
        "--critic_ckpt", 
        type=str, 
        default="ckpt/critic_rl",
        help="Path to the critic checkpoint"
    )
    
    parser.add_argument(
        "--output_filepath", 
        type=str, 
        default="single_sample_output/result.json",
        help="Path to the output JSON file"
    )
    
    args = parser.parse_args()

    generator = SingleSampleGenerator(
        planner_ckpt=args.planner_ckpt,
        critic_ckpt=args.critic_ckpt,
        qwen_processor_ckpt=args.planner_ckpt,  
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    input_image = None

    input_text = "How to draw a cat step by step?"

    # input_text = "<image>. Illustrate a dynamic, step-by-step epic battle between these two characters unleashing their signature cartoon skills. Each step image should containing Cinematic lighting, dramatic camera angles, glowing VFX, flying debris, masterpiece."
    # input_image = ["assets/pikaqiu.png"]

    output_directory = "single_sample_output"
    output_filepath = "single_sample_output/data.json"

    print("\nStarting generation...")
    result_sequence = generator.generate_sequence(
        user_prompt=input_text,
        input_image=input_image,
        output_dir=output_directory,
    )

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(result_sequence, f, ensure_ascii=False, indent=4)