# -*- coding: utf-8 -*-
# Rewards for multimodal tasks with <<answer>...</answer> outputs.
import re
import json
import base64
import io
import os
import random
import copy
import time
import math
import threading
from typing import Any, Dict, List, Optional, Tuple
import json_repair
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from tqdm import tqdm

# ===================== Gemini Scorer Dependencies =====================
import viescore.vie_prompts as vie_prompts
from viescore.utils import mllm_output_to_dict
from google import genai
from google.genai import types
# ======================================================================

# ===================== Gemini Scorer Configuration =====================
# Gemini API key list (supports multiple keys for load balancing)
GEMINI_API_KEYS: List[str] = []

# Gemini configuration parameters
GEMINI_MODEL_NAME: str = "gemini-2.5-pro"
GEMINI_MAX_CLIENT_RETRIES: int = 20
GEMINI_RETRY_DELAY: float = 2.0
GEMINI_MAX_WORKERS: int = int(os.getenv("EDIT_API_MAX_WORKERS", 256))  # Maximum worker threads for parallel calls

# Whether to enable Gemini scoring (if False, will use default placeholder score)
GEMINI_ENABLED: bool = True

# Gemini scorer singleton instance
_GEMINI_SCORER_INSTANCE: Optional["GeminiEditScorer"] = None
# ==========================================================

# ===================== Image Edit API Configuration =====================
# Edit API endpoint configuration (choose one of two methods)
EDIT_API_ENDPOINT: Optional[str] = os.getenv("EDIT_API_ENDPOINT", None)
EDIT_API_ENDPOINT_FILE: Optional[str] = os.getenv("EDIT_API_ENDPOINT_FILE", None) 

# Default edit model name to use
EDIT_MODEL_NAME: str = os.getenv("EDIT_MODEL_NAME", "klein")  # Options: "omnigen2", "klein", "qwen-image-edit"

# Edit API maximum retry count
EDIT_API_MAX_RETRIES: int = 10

# Edit API maximum parallel worker threads
EDIT_API_MAX_WORKERS: int = int(os.getenv("EDIT_API_MAX_WORKERS", 256))

MAX_DURATION: int = int(os.getenv("MAX_DURATION", 70))
# Edit API endpoint pool singleton instance
_EDIT_API_ENDPOINT_POOL: Optional["EndpointPool"] = None

IMAGE_SCALE_WEIGHT: float = float(os.getenv("IMAGE_SCALE_WEIGHT", 1))
GROUP: int = int(os.getenv("GROUP", 8))
# ==========================================================

# -------------------------
# Patterns for format check
# -------------------------
ANSWER_CAPTURE_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL
)

# -------------------------
# Utilities
# -------------------------
def extract_answer(response_text: str) -> Optional[str]:
    try:
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if answer_match:
            json_text = answer_match.group(1).strip()
            json_text = json_repair.loads(json_text)
            return json_text
        else:
            return None
    except:
        return None

def tag_format_reward(response: str) -> float:
    has_reasoning = bool(re.search(r'<', response, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL | re.IGNORECASE))

    if not (has_reasoning and has_answer):
        return 0.0
    
    # Check if the order is correct
    reasoning_match = re.search(r'<think>', response, re.IGNORECASE)
    answer_match = re.search(r'<answer>', response, re.IGNORECASE)
    
    reasoning_pos = reasoning_match.start() if reasoning_match else -1
    answer_pos = answer_match.start() if answer_match else -1
    
    if not (reasoning_pos < answer_pos):
        return 0.0

    answer_dict = extract_answer(response)
    if not answer_dict:
        return 0.0
    
    required_fields = ["previous_step_success", "refine_prompt"]
    has_all_required = all(field in answer_dict for field in required_fields)
    fields_valid = isinstance(answer_dict.get("previous_step_success"), bool) and isinstance(answer_dict.get("refine_prompt"), str)

    return 1.0 if (has_all_required and fields_valid) else 0.0

# -------------------------
# Image Edit API Utility Functions
# -------------------------
class EndpointPool:
    """API endpoint pool, supports round-robin access to multiple endpoints"""
    
    def __init__(self, endpoints: List[str], verbose: bool = False):
        if not endpoints:
            raise ValueError("Endpoint list cannot be empty")
        
        self.endpoints = endpoints
        self.current_index = 0
        self.lock = threading.Lock()
        
        if verbose:
            print(f"📡 Initializing Edit API endpoint pool with {len(self.endpoints)} endpoints:")
            for i, endpoint in enumerate(self.endpoints):
                print(f"  [{i+1}] {endpoint}")
    
    def get_next_endpoint(self) -> str:
        with self.lock:
            endpoint = self.endpoints[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.endpoints)
            return endpoint

def load_endpoints_from_file(file_path: str, default_port: int = 8007, default_protocol: str = "http") -> List[str]:
    endpoints = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('http://') or line.startswith('https://'):
                    endpoints.append(line)
                else:
                    endpoints.append(f"{default_protocol}://{line}:{default_port}")
        if not endpoints:
            print(f"Warning: No valid endpoints read from file {file_path}")
    except Exception as e:
        print(f"Error: Failed to read endpoint file {file_path}: {e}")
    return endpoints

def get_edit_api_endpoint_pool() -> Optional[EndpointPool]:
    global _EDIT_API_ENDPOINT_POOL
    
    if _EDIT_API_ENDPOINT_POOL is not None:
        return _EDIT_API_ENDPOINT_POOL
    
    if EDIT_API_ENDPOINT_FILE:
        endpoints = load_endpoints_from_file(EDIT_API_ENDPOINT_FILE)
        if endpoints:
            _EDIT_API_ENDPOINT_POOL = EndpointPool(endpoints, verbose=True)
            return _EDIT_API_ENDPOINT_POOL
    
    if EDIT_API_ENDPOINT:
        _EDIT_API_ENDPOINT_POOL = EndpointPool([EDIT_API_ENDPOINT], verbose=False)
        return _EDIT_API_ENDPOINT_POOL
    
    print("Warning: Edit API endpoint not configured. Please set EDIT_API_ENDPOINT or EDIT_API_ENDPOINT_FILE")
    return None

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data))
    return image

def get_model_config(model_name: str) -> dict:
    if model_name == "omnigen2":
        return {'num_inference_step': 50, 'text_guidance_scale': 5.0, 'image_guidance_scale': 1.5, 'width': 1024, 'height': 1024, 'negative_prompt': "", 'enhance_prompt': False}
    elif model_name == "klein":
        return {'num_inference_step': 4, 'guidance_scale': 1.0, 'width': 1024, 'height': 1024, 'enhance_prompt': False}
    elif model_name == "qwen-image-edit":
        return {'num_inference_step': 50, 'guidance_scale': 5.0, 'width': 1024, 'height': 1024, 'enhance_prompt': False}
    elif model_name == "longcat-image-edit":
        return {'num_inference_step': 50, 'guidance_scale': 4.5, 'width': 1024, 'height': 1024, 'enhance_prompt': False, 'negative_prompt': ""}
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def call_edit_model_api(
    instruction: str,
    input_image: Image.Image,
    endpoint_pool: Optional[EndpointPool] = None,
    model_config: dict = None,
    max_retries: int = 3,
    total_timeout: Optional[float] = None,
    start_time: Optional[float] = None
) -> Optional[Image.Image]:
    if input_image is not None:
        image_b64 = encode_image_to_base64(input_image)
    else:
        image_b64 = None
    request_data = {
        "image": image_b64,
        "prompt": instruction,
        **(model_config or {}),
    }
    
    total_attempts = max_retries * len(endpoint_pool.endpoints)
    current_endpoint = None
    for attempt in range(total_attempts):
        if total_timeout is not None and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed >= total_timeout:
                return None
            remaining_timeout = max(1.0, total_timeout - elapsed)
        else:
            remaining_timeout = 50
        
        try:
            current_endpoint = endpoint_pool.get_next_endpoint()
            response = requests.post(
                f"{current_endpoint}/edit",
                json=request_data,
                timeout=remaining_timeout
            )
            result = response.json()
            
            if result.get("success"):
                output_image = decode_base64_to_image(result["edited_image"])
                return output_image
            else:
                print(f"Warning: Edit API returned error (endpoint: {current_endpoint}): {result.get('error', 'Unknown error')}")
                if attempt < total_attempts - 1:
                    continue
                    
        except Exception as e:
            endpoint_str = current_endpoint if current_endpoint else "unknown"
            print(f"Warning: Edit API call failed (attempt {attempt + 1}/{total_attempts}, endpoint: {endpoint_str}): {e}")
            if attempt < total_attempts - 1:
                continue
    return None

def img2base64(img, format="PNG"):
    assert isinstance(img, str) or isinstance(img, Image.Image), "img2base64 only accepts str or Image"
    if isinstance(img, str):
        if not os.path.exists(img):
            raise Exception(f"Image path {img} does not exist")
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(img)
        
    img_byte_arr = io.BytesIO()
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(img_byte_arr, format=format)
    return base64.b64encode(img_byte_arr.getvalue()).decode()

class GeminiAIClient:
    def __init__(self, app_key=None, max_tokens=8192, model_name="gemini-2.5-pro"):
        self.app_key = app_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Initialize the official Gemini Client
        self.client = genai.Client(api_key=self.app_key)
        print(f"Initialized official Gemini Client, model_name: {self.model_name}")
    
    def generate_content(self, contents):
        """
        Call the official API to generate content.
        'contents' can be a list containing text strings and types.Part objects.
        """
        retry_times = 0
        response_text = None

        while retry_times < 10:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.max_tokens,
                    )
                )
                response_text = response.text
                if response_text is not None:
                    break
            except Exception as e:
                print(f"Query failed, retry {retry_times + 1}, error: {e}")
                retry_times += 1
                time.sleep(3) 
                
        if response_text is None:
            return "error"
            
        return response_text


class GeminiEditScorer:
    def __init__(
        self,
        api_keys: List[str],
        model_name: str = "gemini-2.5-pro",
        max_client_retries: int = 5,
        retry_delay: float = 3.0
    ) -> None:
        if not api_keys:
            raise ValueError("api_keys must be provided to initialize Gemini scorer.")

        self._clients = [
            GeminiAIClient(app_key=key, max_tokens=4096, model_name=model_name)
            for key in api_keys
        ]
        self._lock = threading.Lock()
        self._next_index = 0
        self._max_client_retries = max(1, max_client_retries)
        self._retry_delay = retry_delay

    def _get_next_client(self) -> "GeminiAIClient":
        with self._lock:
            client = self._clients[self._next_index % len(self._clients)]
            self._next_index += 1
        return client

    def _construct_msgs(self, text: str, b64_imgs: Optional[List[str]]) -> list:
        """
        Construct the contents list accepted by the official SDK.
        """
        contents = [text]
        
        if b64_imgs is not None:
            for img_b64 in b64_imgs:
                img_bytes = base64.b64decode(img_b64)
                image_part = types.Part.from_bytes(
                    data=img_bytes,
                    mime_type='image/png'
                )
                contents.append(image_part)
                
        return contents

    def _call_gemini(self, prompt: str, b64_imgs: Optional[List[str]] = None) -> Optional[str]:
        contents = self._construct_msgs(prompt, b64_imgs)
        
        for attempt in range(self._max_client_retries):
            client = self._get_next_client()
            try:
                response_text = client.generate_content(contents)
                if response_text != "error":
                    return response_text
            except Exception as e:
                print(f"Scorer attempt {attempt + 1} failed: {e}")
                time.sleep(self._retry_delay)
                
        return None

    def score(
        self,
        original_image: Image.Image,
        edited_image: Image.Image,
        instruction: str,
        resize_to_match: bool = True,
        fallback_score: float = 0.0,
    ) -> Tuple[float, Dict[str, Any]]:
        
        if not self._clients:
            return fallback_score, {"error": "no_available_clients"}

        prompt = (instruction or "").strip()
        if not prompt:
            return fallback_score, {"error": "empty_instruction"}

        original_rgb = original_image.convert("RGB")
        edited_rgb = edited_image.convert("RGB")
        if resize_to_match and edited_rgb.size != original_rgb.size:
            edited_rgb = edited_rgb.resize(original_rgb.size)

        # Convert PIL Images to base64 directly in memory
        source_b64 = encode_image_to_base64(original_rgb)
        target_b64 = encode_image_to_base64(edited_rgb)

        # Construct Prompts
        context = vie_prompts._context_no_delimit
        SC_prompt_text = "\n".join([context, vie_prompts._prompts_0shot_two_image_edit_rule_ours, vie_prompts._prompts_0shot_tie_rule_SC_ours])
        PQ_prompt_text = "\n".join([context, vie_prompts._prompts_0shot_rule_PQ_ours])

        _SC_prompt = SC_prompt_text.replace("<instruction>", prompt)

        try:
            SC_dict = False
            PQ_dict = False
            
            max_tries = 1
            tries = 0
            
            while SC_dict is False or PQ_dict is False:
                tries += 1
                guess_if_cannot_parse = True if tries > max_tries else False
                
                if SC_dict is False:
                    result_SC_text = self._call_gemini(
                        prompt=_SC_prompt,
                        b64_imgs=[source_b64, target_b64]
                    )
                    if result_SC_text is None:
                        raise RuntimeError("SC Gemini API call failed.")
                    SC_dict = mllm_output_to_dict(result_SC_text, give_up_parsing=guess_if_cannot_parse)

                if PQ_dict is False:
                    result_PQ_text = self._call_gemini(
                        prompt=PQ_prompt_text,
                        b64_imgs=[target_b64]
                    )
                    if result_PQ_text is None:
                        raise RuntimeError("PQ Gemini API call failed.")
                    PQ_dict = mllm_output_to_dict(result_PQ_text, give_up_parsing=guess_if_cannot_parse)
                
                if tries > max_tries + 2:
                    raise RuntimeError("Failed to parse MLLM output after multiple attempts.")

            if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
                return fallback_score, {"error": "rate_limit_exceeded"}
                
            SC_score = min(SC_dict['score'])
            PQ_score = min(PQ_dict['score'])
            O_score = math.sqrt(SC_score * PQ_score)
            
            return O_score, {
                "semantics": SC_score,
                "quality": PQ_score,
                "overall": O_score,
            }
            
        except Exception as exc:
            print(f"Warning: Gemini evaluation failed: {exc}")
            return fallback_score, {"error": str(exc)}
# ===================================================================

def get_gemini_scorer() -> Optional[GeminiEditScorer]:
    """
    Get Gemini scorer singleton instance
    """
    global _GEMINI_SCORER_INSTANCE
    
    if not GEMINI_ENABLED:
        return None
    
    if not GEMINI_API_KEYS:
        print("Warning: Gemini configuration is incomplete. Please set GEMINI_API_KEYS.")
        return None
    
    if _GEMINI_SCORER_INSTANCE is None:
        try:
            _GEMINI_SCORER_INSTANCE = GeminiEditScorer(
                api_keys=GEMINI_API_KEYS,
                model_name=GEMINI_MODEL_NAME,
                max_client_retries=GEMINI_MAX_CLIENT_RETRIES,
                retry_delay=GEMINI_RETRY_DELAY
            )
            print(f"Gemini scorer initialized with {len(GEMINI_API_KEYS)} key(s) and model: {GEMINI_MODEL_NAME}")
        except Exception as e:
            print(f"Error: Failed to initialize Gemini scorer: {e}")
            _GEMINI_SCORER_INSTANCE = None
    
    return _GEMINI_SCORER_INSTANCE

def evaluate_image_prompt_alignment_gemini(
    original_image: Optional[Image.Image],
    edited_image: Optional[Image.Image],
    instruction: str,
    scorer: Optional[GeminiEditScorer] = None,
    fallback_score: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate image-instruction alignment using Gemini
    """
    if original_image is None or edited_image is None:
        return fallback_score, {"error": "missing_image"}

    if scorer is None:
        scorer = get_gemini_scorer()
        if scorer is None:
            return fallback_score, {"error": "scorer_not_available"}

    return scorer.score(original_image, edited_image, instruction, fallback_score=fallback_score)

def accuracy_reward(skip_image_edit, gt_extracted) -> float:
    if skip_image_edit == -1:
        return 0.0
    if skip_image_edit == gt_extracted:
        return 1.0
    else:
        return 0.0

def get_edited_image_and_score(edit_image_queue, endpoint_pool, edit_model_config, gemini_scorer):

    def call_single_internal(info):
        if info["skip_image_edit"]:
            return info["idx"], None

        start_time = time.time()
        
        instruction = info["refined_prompt"]
        image_path = info["original_image_path"]
        image = Image.open(image_path).convert("RGB")

        h, w = image.size
        if IMAGE_SCALE_WEIGHT != 1:
            image = image.resize((int(h * IMAGE_SCALE_WEIGHT), int(w * IMAGE_SCALE_WEIGHT)))
        print(f"image size {h}, {w} -> {image.size}")
        
        if time.time() - start_time >= MAX_DURATION:
            return info["idx"], None

        item_model_config = copy.deepcopy(edit_model_config)
        if "seed" in info:
            item_model_config["seed"] = info["seed"]
        
        if 'data/interleave/white.png' in image_path:
            edited_image = call_edit_model_api(
                instruction=instruction,
                input_image=None,
                endpoint_pool=endpoint_pool,
                model_config=item_model_config,
                max_retries=EDIT_API_MAX_RETRIES,
                total_timeout=MAX_DURATION,
                start_time=start_time
            )
        else:
            edited_image = call_edit_model_api(
                instruction=instruction,
                input_image=image,
                endpoint_pool=endpoint_pool,
                model_config=item_model_config,
                max_retries=EDIT_API_MAX_RETRIES,
                total_timeout=MAX_DURATION,
                start_time=start_time
            )

        if time.time() - start_time >= MAX_DURATION:
            return info["idx"], None

        if edited_image is None:
            return info["idx"], None

        score, details = evaluate_image_prompt_alignment_gemini(
            original_image=image,
            edited_image=edited_image,
            instruction=info["origin_prompt"],
            scorer=gemini_scorer,
            fallback_score=0.0
        )

        if time.time() - start_time >= MAX_DURATION:
            return info["idx"], None

        info["gpt_score"] = score
        info["gpt_detail"] = details
        return info["idx"], info

    def call_single(info):
        single_executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = single_executor.submit(call_single_internal, info)
            try:
                result = future.result(timeout=MAX_DURATION)
                return result
            except FutureTimeoutError:
                future.cancel()
                print(f"Func get_edited_image_and_score() Error: Sample {info['idx']} timed out ({MAX_DURATION} seconds)")
                return info["idx"], None
            except Exception as e:
                future.cancel()
                print(f"Func get_edited_image_and_score() Error: Sample {info['idx']} processing exception: {e}")
                return info["idx"], None
            finally:
                single_executor.shutdown(wait=False)
        except Exception as e:
            print(f"Func get_edited_image_and_score() Error: Sample {info['idx']} executor exception: {e}")
            return info["idx"], None

    with ThreadPoolExecutor(max_workers=GEMINI_MAX_WORKERS) as executor:
        futures = {
            executor.submit(call_single, info): info["idx"]
            for info in edit_image_queue
        }
        with tqdm(total=len(edit_image_queue), desc="Processing image editing and scoring", unit="sample") as pbar:
            for future in as_completed(futures):
                try:
                    idx, info = future.result()
                    edit_image_queue[idx] = info
                    pbar.update(1)
                except Exception as e:
                    idx = futures[future]
                    print(f"Func get_edited_image_and_score() Error: Sample {idx} processing exception: {e}")
                    edit_image_queue[idx] = None
                    pbar.update(1)

    return edit_image_queue

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.5,
) -> List[Dict[str, float]]:
    results = []
    image_edit_queue = [] 
    for idx, item in enumerate(reward_inputs):
        try:
            raw_response = item["response"]
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)
            gt_extracted = item["ground_truth"]

            format_reward = tag_format_reward(response)
            answer_all = extract_answer(response) or {}

            refined_prompt = answer_all.get('refine_prompt', "")
            pred_judge = answer_all.get('previous_step_success', -1)

            previous_image_score = (gt_extracted["semantics"] * gt_extracted["quality"]) ** 0.5

            skip_image_edit = gt_extracted["success"]

            judge_accuracy_reward = accuracy_reward(pred_judge, gt_extracted["success"])
            
            origin_prompt = item["origin_prompt"]
            origin_image_path = item.get("origin_image_path", None)
            
            # if idx % GROUP == 0:
            #     seed = 0
              
            image_edit_queue.append({
                "idx": idx,
                "original_image_path": origin_image_path,
                "origin_prompt": origin_prompt, 
                "refined_prompt": refined_prompt if refined_prompt!="" else item['previous_prompt'],
                "judge_accuracy_reward": judge_accuracy_reward, 
                "previous_image_score": previous_image_score,
                "previous_image_semantic_score": gt_extracted["semantics"],
                "previous_image_quality_score": gt_extracted["quality"],
                "seed": 0,
                "skip_image_edit": skip_image_edit,
            })

            results.append({
                "overall": 0.0, 
                "format_reward": float(format_reward), 
                "judge_accuracy_reward": judge_accuracy_reward, 
                "edited_image_reward_semantic": 0.0,
                "edited_image_reward_quality": 0.0,
                "idx": idx,
            })
        except Exception as e:
            print(f"Func compute_score() Error: Sample {idx} processing exception: {e}")
            results.append({"overall": 0.0, "format_reward": 0.0, "judge_accuracy_reward": 0.0, "edited_image_reward": 0.0})

    # Get edited images and scores
    edit_endpoint_pool = get_edit_api_endpoint_pool()
    edit_model_config = get_model_config(EDIT_MODEL_NAME)
    
    gemini_scorer = get_gemini_scorer()
            
    image_edit_results = get_edited_image_and_score(image_edit_queue, edit_endpoint_pool, edit_model_config, gemini_scorer)

    for idx, result in enumerate(image_edit_results):

        if result is None:
            results[idx]['edited_image_reward_semantic'] = 0.5
            results[idx]['edited_image_reward_quality'] = 0.5
            results[idx]['overall'] = 0.5 * results[idx]['format_reward'] + 0.5 * (0.2 * results[idx]['judge_accuracy_reward'] + 0.6 * results[idx]['edited_image_reward_semantic'] + 0.2 * results[idx]['edited_image_reward_quality'])
            continue

        edited_image_semantic_score = result['gpt_detail']['semantics']
        edited_image_quality_score = result['gpt_detail']['quality']
        
        edited_image_reward_semantic = edited_image_semantic_score - result['previous_image_semantic_score']
        edited_image_reward_quality = edited_image_quality_score - result['previous_image_quality_score']
        
        edited_image_reward_semantic = edited_image_reward_semantic / 10.0
        edited_image_reward_semantic = (edited_image_reward_semantic + 1.0) / 2.0
        edited_image_reward_quality = edited_image_reward_quality / 10.0
        edited_image_reward_quality = (edited_image_reward_quality + 1.0) / 2.0
        
        results[idx]['edited_image_reward_semantic'] = edited_image_reward_semantic
        results[idx]['edited_image_reward_quality'] = edited_image_reward_quality
        
        results[idx]['overall'] = 0.5 * results[idx]['format_reward'] + 0.5 * (0.2 * results[idx]['judge_accuracy_reward'] + 0.6 * edited_image_reward_semantic + 0.2 * edited_image_reward_quality)

    return results