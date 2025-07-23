import time
import io, os, sys
from flask_cors import CORS
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import numpy as np
from flask import Flask, request, Response
import torch
import torchaudio
from modelscope import snapshot_download
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

if not os.path.exists('pretrained_models/CosyVoice-300M/cosyvoice.yaml') or not os.path.exists('pretrained_models/CosyVoice-300M-SFT/cosyvoice.yaml'):
    snapshot_download('iic/CosyVoice-300M', cache_dir='pretrained_models/CosyVoice-300M',local_dir='pretrained_models/CosyVoice-300M')
    snapshot_download('iic/CosyVoice-300M-SFT', cache_dir='pretrained_models/CosyVoice-300M-SFT',local_dir='pretrained_models/CosyVoice-300M-SFT')

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')

print(cosyvoice.list_available_spks())

app = Flask(__name__)
CORS(app)  

@app.route("/inference/stream", methods=['POST'])
def stream():
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
        for chunk in cosyvoice.stream(tts_text, prompt_text, prompt_speech_16k):
            float_data = chunk.numpy()
            byte_data = float_data.tobytes()
            print(f"len data: {len(byte_data)}")
            yield byte_data

    return Response(generate_stream(), mimetype="audio/pcm") 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080,)
