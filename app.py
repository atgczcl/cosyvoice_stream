import time
import io
import os
import sys
from flask_cors import CORS
import numpy as np
from flask import Flask, request, Response
import torch
import torchaudio
from modelscope import snapshot_download
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import json
import base64
import threading
import uuid
from cosyvoice.utils.file_utils import logging
from typing import Dict, Any, Generator, Tuple

# 其他导入和初始化代码保持不变
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

if not os.path.exists('pretrained_models/CosyVoice-300M/cosyvoice.yaml') or not os.path.exists('pretrained_models/CosyVoice-300M-SFT/cosyvoice.yaml'):
    snapshot_download('iic/CosyVoice-300M', cache_dir='pretrained_models/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
    snapshot_download('iic/CosyVoice-300M-SFT', cache_dir='pretrained_models/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

print(cosyvoice.list_avaliable_spks())

app = Flask(__name__)
CORS(app)

# 缓存所有请求的 stop_generation_flag
stop_generation_flags = {}

def get_stop_generation_flag(request_id=None) -> Tuple[threading.Event, str]:
    """为每个请求生成独立的Event实例，并缓存到全局字典中"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    if request_id not in stop_generation_flags:
        stop_generation_flags[request_id] = threading.Event()
    return stop_generation_flags[request_id], request_id

@app.route("/inference/streamclone", methods=['POST'])
def streamclone():
    stop_generation_flag, request_id = get_stop_generation_flag()
    question_data = request.get_json()
    tts_text = question_data.get('query')
    prompt_text = question_data.get('prompt_text')
    prompt_speech = load_wav(question_data.get('prompt_speech'), 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)
    if not tts_text:
        return {"error": "Query parameter 'query' is required"}, 400

    def generate_stream():
        for chunk in cosyvoice.stream_clone(tts_text, prompt_text, prompt_speech_16k, 1.0, stop_generation_flag):
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

    return Response(generate_stream(), mimetype="text/event-stream"), 200, {'X-Request-ID': request_id}

@app.route("/inference/stream_sft", methods=['POST'])
def stream_sft():
    stop_generation_flag, request_id = get_stop_generation_flag()
    question_data = request.get_json()
    query = question_data.get('query')
    speaker = question_data.get('speaker')
    if not query:
        return {"error": "Query parameter 'query' is required"}, 400

    def generate_stream():
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

    return Response(generate_stream(), mimetype="audio/pcm"), 200, {'X-Request-ID': request_id}

@app.route("/inference/stream_sft_json", methods=['POST'])
def stream_sft_json():
    stop_generation_flag, request_id = get_stop_generation_flag()
    question_data = request.get_json(silent=True)
    if not question_data:
        return {"error": "Invalid JSON request"}, 400
    
    query = question_data.get('query')
    speaker = question_data.get('speaker')
    speed = question_data.get('speed', 1.0)
    isStream = question_data.get('isStream', False)

    try:
        speed = float(speed)  # 确保速度为浮点数
    except ValueError:
        return {"error": "Invalid value for 'speed'"}, 400

    isStream = isStream == 'true'  # 确保 isStream 为布尔类型

    if not query or not isinstance(query, str):
        return {"error": "Query parameter 'query' must be a non-empty string"}, 400

    def generate_stream():
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

    return Response(generate_stream(), mimetype='text/event-stream'), 200, {'X-Request-ID': request_id}

@app.route("/inference/stop_generation", methods=['POST'])
def stop_generation():
    # 接收 JSON 数据中的 request_ids 列表
    request_ids = request.get_json().get('request_ids', [])
    logging.info(f"Received stop generation request for request IDs: {request_ids}")
    if not request_ids:
        return {"error": "No request IDs provided"}, 400

    stopped_request_ids = []
    for request_id in request_ids:
        if request_id in stop_generation_flags:
            stop_generation_flags[request_id].set()
            del stop_generation_flags[request_id]
            stopped_request_ids.append(request_id)
            logging.info(f"REQStop generation for request ID: {request_id}|{len(stop_generation_flags)}")

    if stopped_request_ids:
        return {"message": f"Stopped generation for request IDs: {stopped_request_ids}"}, 200
    else:
        return {"error": "No active generations found for the provided request IDs"}, 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)