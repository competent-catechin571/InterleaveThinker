#!/usr/bin/env python3
"""
Kelvin API Server
GPU manager based on api_server.py, providing image editing REST API interfaces
"""

import os
import sys
import time
import base64
import io
import json
import random
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Event
import atexit
import signal
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# FastAPI related imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from PIL import Image

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

from diffusers import Flux2KleinPipeline
import torch

# Configuration parameters
model_repo_id = "black-forest-labs/FLUX.2-klein-9B"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

NUM_GPUS_TO_USE = int(os.environ.get("NUM_GPUS_TO_USE", torch.cuda.device_count()))  
TASK_QUEUE_SIZE = int(os.environ.get("TASK_QUEUE_SIZE", 100))  
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 300))

print(f"Configuration info: Using {NUM_GPUS_TO_USE} GPUs, queue size {TASK_QUEUE_SIZE}, timeout {TASK_TIMEOUT} seconds")

def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype, device))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    from safetensors import safe_open
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict

# ============== GPU Manager Class (Adapted for Image Editing) ==============
class EditGPUWorker:
    def __init__(self, gpu_id, model_repo_id, task_queue, result_queue, stop_event):
        self.gpu_id = gpu_id
        self.model_repo_id = model_repo_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        self.pipe = None
        
    def initialize_model(self):
        """Initialize the image editing model on the specified GPU"""
        try:
            print(f"GPU {self.gpu_id} preparing to load model, waiting {self.gpu_id * 2} seconds to stagger...")
            time.sleep(self.gpu_id * 2) 
            
            torch.cuda.set_device(self.gpu_id)
            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            self.pipe = Flux2KleinPipeline.from_pretrained(self.model_repo_id, torch_dtype=torch_dtype)
            # self.pipe = Flux2KleinPipeline.from_pretrained(self.model_repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False)
            self.pipe.to(self.device)
            self.pipe.set_progress_bar_config(disable=True)

            # self.pipe.set_progress_bar_config(disable=None)
            print(f"GPU {self.gpu_id} image generation/editing model initialized successfully")
            return True
        except Exception as e:
            print(f"GPU {self.gpu_id} image generation/editing model initialization failed: {e}")
            return False
    
    def process_task(self, task):
        """Process a single editing task"""
        try:
            task_id = task['task_id']
            image_bytes = task['image_bytes']  # Byte data
            if image_bytes is not None:
                image = bytes_to_image(image_bytes)  # Convert to PIL Image
            else:
                image = None
            prompt = task['prompt']
            negative_prompt = task['negative_prompt']
            seed = task['seed']
            num_inference_steps = task['num_inference_steps']
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            with torch.cuda.device(self.gpu_id):
                with torch.inference_mode():
                    if image is not None:
                        output_image = self.pipe(
                            image=image,
                            prompt=prompt,
                            height=1024,
                            width=1024,
                            guidance_scale=1.0,
                            num_inference_steps=4,
                            generator=generator,
                        ).images[0]
                    else:
                        output_image = self.pipe(
                            prompt=prompt,
                            height=1024,
                            width=1024,
                            guidance_scale=1.0,
                            num_inference_steps=4,
                            generator=generator,
                        ).images[0]
            
            return {
                'task_id': task_id,
                'image': output_image,
                'success': True,
                'gpu_id': self.gpu_id
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'gpu_id': self.gpu_id
            }
    
    def run(self):
        """Worker main loop"""
        if not self.initialize_model():
            return
        
        print(f"GPU {self.gpu_id} edit worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get task from task queue, set timeout to check stop event
                task = self.task_queue.get(timeout=1)
                if task is None:  # Poison pill, exit signal
                    break
                
                # Process task
                result = self.process_task(task)
                
                # Put result into result queue
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU {self.gpu_id} edit worker exception: {e}")
                continue
        
        print(f"GPU {self.gpu_id} edit worker stopped")


# Global GPU worker function, used for spawn mode
def edit_gpu_worker_process(gpu_id, model_repo_id, task_queue, result_queue, stop_event):
    worker = EditGPUWorker(gpu_id, model_repo_id, task_queue, result_queue, stop_event)
    worker.run()


class MultiGPUEditManager:
    def __init__(self, model_repo_id, num_gpus=None, task_queue_size=100):
        self.model_repo_id = model_repo_id
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.task_queue = Queue(maxsize=task_queue_size)  
        self.result_queue = Queue()  
        self.stop_event = Event()
        self.workers = []
        self.worker_processes = []
        self.task_counter = 0
        self.pending_tasks = {}  
        self.pending_tasks_lock = threading.Lock()
        
        print(f"Initializing multi-GPU image editing manager, using {self.num_gpus} GPUs, queue size {task_queue_size}")
        
    def start_workers(self):
        """Start all GPU workers"""
        for gpu_id in range(self.num_gpus):
            process = Process(target=edit_gpu_worker_process, 
                            args=(gpu_id, self.model_repo_id, self.task_queue, 
                                  self.result_queue, self.stop_event))
            process.start()
            
            self.worker_processes.append(process)
        
        # Start result processing thread
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        print(f"All {self.num_gpus} GPU edit workers have started")
    
    def _process_results(self):
        """Background thread to process results"""
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=1)
                task_id = result['task_id']
                
                event_to_set = None
                with self.pending_tasks_lock:
                    if task_id in self.pending_tasks:
                        # Pass the result to the waiting task
                        self.pending_tasks[task_id]['result'] = result
                        event_to_set = self.pending_tasks[task_id]['event']
                if event_to_set:
                    event_to_set.set()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Result processing thread exception: {e}")
                continue
    
    def submit_task(self, image, prompt, negative_prompt=" ", seed=42, 
                   guidance_scale=4.0, num_inference_steps=50, timeout=300):
        """Submit edit task and wait for result"""
        with self.pending_tasks_lock:
            task_id = f"edit_task_{self.task_counter}_{time.time()}"
            self.task_counter += 1
        
        # Convert PIL image to byte data for cross-process transfer
        if image is not None:
            image_bytes = image_to_bytes(image) if hasattr(image, 'save') else image
        else:
            image_bytes = None
        
        task = {
            'task_id': task_id,
            'image_bytes': image_bytes,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
        }
        
        # Create wait event
        result_event = threading.Event()
        with self.pending_tasks_lock:
            self.pending_tasks[task_id] = {
                'event': result_event,
                'result': None,
                'submitted_time': time.time()
            }
        
        try:
            # Put task into queue
            self.task_queue.put(task, timeout=10)
            
            # Wait for result
            start_time = time.time()
            if result_event.wait(timeout=timeout):
                with self.pending_tasks_lock:
                    result = self.pending_tasks.get(task_id, {}).get('result')
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
                return result if result is not None else {'success': False, 'error': 'Unknown error'}
            else:
                # Timeout
                with self.pending_tasks_lock:
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
                return {'success': False, 'error': 'Task timeout'}
                
        except queue.Full:
            with self.pending_tasks_lock:
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
            return {'success': False, 'error': 'Task queue full'}
        except Exception as e:
            with self.pending_tasks_lock:
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
            return {'success': False, 'error': str(e)}
    
    def get_queue_status(self):
        """Get queue status"""
        with self.pending_tasks_lock:
            pending_count = len(self.pending_tasks)
        return {
            'task_queue_size': self.task_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'pending_tasks': pending_count,
            'active_workers': len(self.worker_processes),
            'total_gpus': self.num_gpus
        }
    
    def stop(self):
        """Stop all workers"""
        print("Stopping multi-GPU edit manager...")
        self.stop_event.set()
        
        # Send stop signal to each worker
        for _ in range(self.num_gpus):
            try:
                self.task_queue.put(None, timeout=1)
            except queue.Full:
                pass
        
        # Wait for all processes to finish
        for process in self.worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        print("Multi-GPU edit manager stopped")


# ============== API Related Classes and Functions ==============

# Global GPU manager instance
gpu_manager = None

def initialize_gpu_manager():
    """Initialize global GPU manager"""
    global gpu_manager
    if gpu_manager is None:
        try:
            if torch.cuda.is_available():
                print(f"Detected {torch.cuda.device_count()} GPUs")
            
            gpu_manager = MultiGPUEditManager(
                model_repo_id, 
                num_gpus=NUM_GPUS_TO_USE,
                task_queue_size=TASK_QUEUE_SIZE
            )
            gpu_manager.start_workers()
            print("GPU edit manager initialized successfully")
        except Exception as e:
            print(f"GPU edit manager initialization failed: {e}")
            gpu_manager = None

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL image"""
    buffer = io.BytesIO(base64.b64decode(base64_str))
    return Image.open(buffer)

def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL image to byte data (for cross-process transfer)"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Convert byte data to PIL image"""
    buffer = io.BytesIO(image_bytes)
    return Image.open(buffer)

def process_uploaded_image(file_content: bytes) -> Image.Image:
    """Process uploaded image file"""
    image = Image.open(io.BytesIO(file_content))
    # Convert to RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Limit image size
    width, height = image.size
    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
        ratio = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Image resized to {new_width}x{new_height}")
    
    return image


# ============== Pydantic Models ==============

class ImageEditRequest(BaseModel):
    image: Optional[str] = Field(None, description="Input image (base64 encoded), mutually exclusive with image_file")
    prompt: str = Field(..., description="Edit instruction")
    negative_prompt: Optional[str] = Field(" ", description="Negative prompt")
    seed: Optional[int] = Field(None, description="Random seed, randomly generated if not provided")
    guidance_scale: Optional[float] = Field(4.0, ge=0.0, le=7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, ge=1, le=100, description="Number of inference steps")
    enhance_prompt: Optional[bool] = Field(True, description="Whether to enhance edit prompt")

class ImageEditResponse(BaseModel):
    success: bool = Field(..., description="Whether successful")
    task_id: Optional[str] = Field(None, description="Task ID")
    original_image: Optional[str] = Field(None, description="Original image (base64 encoded)")
    edited_image: Optional[str] = Field(None, description="Edited image (base64 encoded)")
    seed: Optional[int] = Field(None, description="Used random seed")
    original_prompt: Optional[str] = Field(None, description="Original edit instruction")
    enhanced_prompt: Optional[str] = Field(None, description="Enhanced edit instruction")
    gpu_id: Optional[int] = Field(None, description="Used GPU ID")
    processing_time: Optional[float] = Field(None, description="Processing time (seconds)")
    error: Optional[str] = Field(None, description="Error message")

class SystemStatus(BaseModel):
    active_workers: int = Field(..., description="Active worker processes count")
    task_queue_size: int = Field(..., description="Task queue size")
    result_queue_size: int = Field(..., description="Result queue size")
    pending_tasks: int = Field(..., description="Pending tasks count")
    total_gpus: int = Field(..., description="Total GPU count")
    system_ready: bool = Field(..., description="Whether system is ready")


# ============== FastAPI Application ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Initialize GPU manager on startup
    print("Starting image editing API server...")
    initialize_gpu_manager()
    if gpu_manager is None:
        print("Warning: GPU manager initialization failed, some features may be unavailable")
    else:
        print("Image editing API server startup complete")
    
    yield
    
    # Clean up resources on shutdown
    print("Shutting down image editing API server...")
    if gpu_manager:
        gpu_manager.stop()
    print("Image editing API server shut down")

app = FastAPI(
    title="Kelvin API Server",
    description="Image editing API service based on the Kelvin model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Kelvin API Server", "status": "Running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "Healthy",
        "gpu_manager_ready": gpu_manager is not None,
        "timestamp": time.time()
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status"""
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    status = gpu_manager.get_queue_status()
    return SystemStatus(
        active_workers=status['active_workers'],
        task_queue_size=status['task_queue_size'],
        result_queue_size=status['result_queue_size'],
        pending_tasks=status['pending_tasks'],
        total_gpus=status['total_gpus'],
        system_ready=True
    )

@app.post("/edit", response_model=ImageEditResponse)
def edit_image(request: ImageEditRequest):
    """Edit image - using base64 format"""
    
    # Check GPU manager
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    # if not request.image:
    #     raise HTTPException(status_code=400, detail="Input image must be provided")
    
    start_time = time.time()
    
    try:
        # Decode input image
        if request.image is not None:
            input_image = base64_to_image(request.image)
            original_image_b64 = request.image
        else:
            input_image = None
            original_image_b64 = None
        
        # Process seed
        if request.seed is None:
            seed = random.randint(0, MAX_SEED)
        else:
            seed = request.seed
        
        # Process edit prompt
        original_prompt = request.prompt
        enhanced_prompt = original_prompt
        
        # Submit task to GPU queue
        result = gpu_manager.submit_task(
            image=input_image,
            prompt=enhanced_prompt,
            negative_prompt=request.negative_prompt,
            seed=seed,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            timeout=TASK_TIMEOUT,
        )
        
        if result['success']:
            # Convert edited image to base64
            edited_image_b64 = image_to_base64(result['image'])
            processing_time = time.time() - start_time
            
            print(f"Image edited successfully, using GPU {result['gpu_id']}, took {processing_time:.2f} seconds")
            
            return ImageEditResponse(
                success=True,
                task_id=result['task_id'],
                original_image=original_image_b64,
                edited_image=edited_image_b64,
                seed=seed,
                original_prompt=original_prompt,
                enhanced_prompt=enhanced_prompt if request.enhance_prompt else None,
                gpu_id=result['gpu_id'],
                processing_time=processing_time
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Image editing failed: {result['error']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Exception occurred while editing image: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/edit-upload", response_model=ImageEditResponse)
def edit_image_upload(
    image_file: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Edit instruction"),
    negative_prompt: str = Form(" ", description="Negative prompt"),
    seed: Optional[int] = Form(None, description="Random seed"),
    guidance_scale: float = Form(4.0, description="Guidance scale"),
    num_inference_steps: int = Form(50, description="Number of inference steps"),
    enhance_prompt: bool = Form(True, description="Whether to enhance edit prompt")
):
    """Edit image - using file upload format"""
    
    # Check GPU manager
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    # Validate file type
    if not image_file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image format")
    
    start_time = time.time()
    
    try:
        # Read uploaded image file
        file_content = image_file.file.read()
        input_image = process_uploaded_image(file_content)
        
        # Convert original image to base64 for return
        original_image_b64 = image_to_base64(input_image)
        
        # Process seed
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        
        enhanced_prompt = prompt
        
        # Submit task to GPU queue
        result = gpu_manager.submit_task(
            image=input_image,
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            timeout=TASK_TIMEOUT,
        )
        
        if result['success']:
            # Convert edited image to base64
            edited_image_b64 = image_to_base64(result['image'])
            processing_time = time.time() - start_time
            
            print(f"Image edited successfully, using GPU {result['gpu_id']}, took {processing_time:.2f} seconds")
            
            return ImageEditResponse(
                success=True,
                task_id=result['task_id'],
                original_image=original_image_b64,
                edited_image=edited_image_b64,
                seed=seed,
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt if enhance_prompt else None,
                gpu_id=result['gpu_id'],
                processing_time=processing_time
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Image editing failed: {result['error']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Exception occurred while editing image: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


# ============== Startup Script ==============

def cleanup():
    """Cleanup function"""
    if gpu_manager:
        gpu_manager.stop()

if __name__ == "__main__":
    def signal_handler(signum, frame):
        print(f"Received signal {signum}, cleaning up resources...")
        cleanup()
        exit(0)
    
    # Register cleanup function
    atexit.register(cleanup)
    
    # Handle signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start server
        # Read port from environment variable, use default 8011 if not available
        port = int(os.getenv("SERVICE_PORT", "8011"))
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,  # Qwen Image Edit Plus port
            workers=1,  # Must be 1 because we use a multi-process GPU manager
            log_level="info"
        )
    except KeyboardInterrupt:
        print("Received interrupt signal, cleaning up resources...")
        cleanup()
    except Exception as e:
        print(f"Application exception: {e}")
        cleanup()
        raise