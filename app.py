
import os
import sys
import numpy as np
import torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from modelscope import snapshot_download

import json
import base64
import threading
import uuid
from typing import Dict, Any, Generator, Tuple



if  not os.path.exists('pretrained_models/CosyVoice2-0.5B/cosyvoice.yaml'):
    # or not os.path.exists('pretrained_models/CosyVoice-300M/cosyvoice.yaml'):
    # snapshot_download('iic/CosyVoice-300M', cache_dir='pretrained_models/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
    snapshot_download('iic/CosyVoice2-0.5B', cache_dir='pretrained_models/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

# cosyvoice = CosyVoice('pretrained_models/CosyVoice2-0.5B')
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

print(cosyvoice.list_available_spks())

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 缓存所有请求的 stop_generation_flag
stop_generation_flags = {}

def get_stop_generation_flag(request_id=None) -> Tuple[threading.Event, str]:
    """为每个请求生成独立的Event实例，并缓存到全局字典中"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    if request_id not in stop_generation_flags:
        stop_generation_flags[request_id] = threading.Event()
    return stop_generation_flags[request_id], request_id

@app.post("/inference/streamclone")
async def streamclone(request: Request):
    stop_generation_flag, request_id = get_stop_generation_flag()
    question_data = await request.json()
    tts_text = question_data.get('query')
    prompt_text = question_data.get('prompt_text')
    prompt_speech = load_wav(question_data.get('prompt_speech'), 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)
    if not tts_text:
        raise HTTPException(status_code=400, detail="Query parameter 'query' is required")

    async def generate_stream():
        for chunk in cosyvoice.stream_clone(tts_text, prompt_text, prompt_speech_16k, True, 1.0, stop_generation_flag):
            if stop_generation_flag.is_set():
                break
            float_data = chunk.numpy()
            byte_data = float_data.tobytes()
            logging.info(f"len data: {len(byte_data)}")
            encoded_data = base64.b64encode(byte_data).decode('utf-8')
            json_data = {"data": encoded_data}
            yield f"{json.dumps(json_data)}\n\n"
        
        # 数据发送完成后设置 stop_generation_flag 为 True 并删除 request_id
        with threading.Lock():
            stop_generation_flag.set()
            if request_id in stop_generation_flags:
                del stop_generation_flags[request_id]
                logging.info(f"Auto Stop generation for request ID: {request_id}")

    rsp = Response(generate_stream(), media_type="text/event-stream")
    rsp.headers['X-Request-ID'] = request_id
    rsp.headers['Access-Control-Expose-Headers'] = 'X-Request-ID'
    return rsp

@app.post("/inference/stream_sft")
async def stream_sft(request: Request):
    stop_generation_flag, request_id = get_stop_generation_flag()
    question_data = await request.json()
    query = question_data.get('query')
    speaker = question_data.get('speaker')
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter 'query' is required")

    async def generate_stream():
        for chunk in cosyvoice.stream_sft(query, speaker, True, 1.0, stop_generation_flag):
            if stop_generation_flag.is_set():
                break
            float_data = chunk.numpy()
            byte_data = float_data.tobytes()
            logging.info(f"len data: {len(byte_data)}")
            yield byte_data
        
        # 数据发送完成后设置 stop_generation_flag 为 True 并删除 request_id
        with threading.Lock():
            stop_generation_flag.set()
            if request_id in stop_generation_flags:
                del stop_generation_flags[request_id]
                logging.info(f"Auto Stop generation for request ID: {request_id}")

    rsp = Response(generate_stream(), media_type="audio/pcm")
    rsp.headers['X-Request-ID'] = request_id
    rsp.headers['Access-Control-Expose-Headers'] = 'X-Request-ID'
    return rsp

@app.post("/inference/stream_sft_json")
async def stream_sft_json(request: Request):
    stop_generation_flag, request_id = get_stop_generation_flag()
    try:
        question_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request")
    
    query = question_data.get('query')
    speaker = question_data.get('speaker')
    speed = question_data.get('speed', 1.0)
    isStream = question_data.get('isStream', False)

    try:
        speed = float(speed)  # 确保速度为浮点数
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid value for 'speed'")

    isStream = isStream == 'true'  # 确保 isStream 为布尔类型

    if not query or not isinstance(query, str):
        raise HTTPException(status_code=400, detail="Query parameter 'query' must be a non-empty string")

    async def generate_stream():
        for chunk in cosyvoice.stream_sft(query, speaker, isStream, speed, stop_generation_flag):
            if stop_generation_flag.is_set():
                break
            float_data = chunk.numpy()
            byte_data = float_data.tobytes()
            logging.info(f"len data: {len(byte_data)}")
            encoded_data = base64.b64encode(byte_data).decode('utf-8')
            json_data = {"data": encoded_data}
            yield f"{json.dumps(json_data)}\n\n"
        
        # 数据发送完成后设置 stop_generation_flag 为 True 并删除 request_id
        with threading.Lock():
            stop_generation_flag.set()
            if request_id in stop_generation_flags:
                del stop_generation_flags[request_id]
                logging.info(f"Auto Stop generation for request ID: {request_id}|{len(stop_generation_flags)}")
    
    rsp = Response(generate_stream(), media_type="text/event-stream")
    rsp.headers['X-Request-ID'] = request_id
    rsp.headers['Access-Control-Expose-Headers'] = 'X-Request-ID'
    return rsp

@app.post("/inference/stop_generation")
async def stop_generation(request: Request):
    # 接收 JSON 数据中的 request_ids 列表
    request_ids = (await request.json()).get('request_ids', [])
    logging.info(f"Received stop generation request for request IDs: {request_ids}")
    if not request_ids:
        raise HTTPException(status_code=400, detail="No request IDs provided")

    stopped_request_ids = []
    for request_id in request_ids:
        if request_id in stop_generation_flags:
            stop_generation_flags[request_id].set()
            del stop_generation_flags[request_id]
            stopped_request_ids.append(request_id)
            logging.info(f"REQStop generation for request ID: {request_id}|{len(stop_generation_flags)}")

    if stopped_request_ids:
        return {"message": f"Stopped generation for request IDs: {stopped_request_ids}"}
    else:
        raise HTTPException(status_code=404, detail="No active generations found for the provided request IDs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)