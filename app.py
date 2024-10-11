import time
import io, os, sys
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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

if not os.path.exists('pretrained_models/CosyVoice-300M/cosyvoice.yaml') or not os.path.exists('pretrained_models/CosyVoice-300M-SFT/cosyvoice.yaml'):
    snapshot_download('iic/CosyVoice-300M', cache_dir='pretrained_models/CosyVoice-300M',local_dir='pretrained_models/CosyVoice-300M')
    snapshot_download('iic/CosyVoice-300M-SFT', cache_dir='pretrained_models/CosyVoice-300M-SFT',local_dir='pretrained_models/CosyVoice-300M-SFT')

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

print(cosyvoice.list_avaliable_spks())
# 创建一个线程安全的事件对象，用于控制生成过程的停止
stop_generation_flag = threading.Event()
app = Flask(__name__)
CORS(app)  

@app.route("/inference/streamclone", methods=['POST'])
def streamclone():
    stop_generation_flag.clear()
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
            print(f"len data: {len(byte_data)}")
            yield byte_data

    return Response(generate_stream(), mimetype="audio/pcm") 

@app.route("/inference/stream_sft", methods=['POST'])
def stream_sft():
    stop_generation_flag.clear()
    question_data = request.get_json()
    query = question_data.get('query')
    speaker = question_data.get('speaker')
    if not query:
        return {"error": "Query parameter 'query' is required"}, 400

    def generate_stream():
        for chunk in cosyvoice.stream_sft(query, speaker, True, 1.0, stop_generation_flag):
            if stop_generation_flag.is_set():
                break
            # yield chunk.numpy().tobytes()  # Assuming chunk is a PyTorch tensor
            float_data = chunk.numpy()
            byte_data = float_data.tobytes()
            print(f"len data: {len(byte_data)}")
            yield byte_data

    return Response(generate_stream(), mimetype="audio/pcm")  # Custom mimetype for pcm data

@app.route("/inference/stream_sft_json", methods=['POST'])
def stream_sft_json():
    stop_generation_flag.clear()
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
            # 将字节数据转换为Base64编码的字符串
            encoded_data = base64.b64encode(byte_data).decode('utf-8')
            json_data = {"data": encoded_data}
            yield f"{json.dumps(json_data)}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')  # Custom mimetype for pcm data


@app.route("/inference/stop_generation", methods=['POST'])
def stop_generation():
    # 停止生成过程
    stop_generation_flag.set()
    print("Stop generation.")
    return {"message": "Stop generation successfully."}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080,)
