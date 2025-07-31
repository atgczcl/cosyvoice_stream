import os
import sys
import numpy as np
import torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PreDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 添加项目根目录到Python路径
sys.path.append(PreDir)
# 确保正确添加Matcha-TTS路径
sys.path.append(os.path.join(PreDir, 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# 修改: 使用 StreamingResponse 来处理异步生成器
from fastapi.responses import StreamingResponse
import numpy as np
from modelscope import snapshot_download

import json
import base64
import threading
import uuid
from typing import Dict, Any, Generator, Tuple
import re
import time
import torchaudio

#ROOT_DIR 的父目录路径

print('loading model...', ROOT_DIR, os.path.join(PreDir, 'pretrained_models/CosyVoice2-0.5B'))
# 导入模型 在文件夹上层目录下
if not os.path.exists(os.path.join(PreDir, 'pretrained_models/CosyVoice2-0.5B/cosyvoice2.yaml')):
    snapshot_download('iic/CosyVoice2-0.5B', cache_dir=os.path.join(PreDir, 'pretrained_models/CosyVoice2-0.5B'), local_dir=os.path.join(PreDir, 'pretrained_models/CosyVoice2-0.5B'))

cosyvoice = CosyVoice2(os.path.join(PreDir, 'pretrained_models/CosyVoice2-0.5B'), load_jit=False, load_trt=False, load_vllm=False, fp16=False)


# if  not os.path.exists('/pretrained_models/CosyVoice2-0.5B/cosyvoice2.yaml'):
#     snapshot_download('iic/CosyVoice2-0.5B', cache_dir='../pretrained_models/CosyVoice2-0.5B', local_dir='../pretrained_models/CosyVoice2-0.5B')
    # or not os.path.exists('pretrained_models/CosyVoice-300M/cosyvoice.yaml'):
    # snapshot_download('iic/CosyVoice-300M', cache_dir='pretrained_models/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')

# cosyvoice = CosyVoice('pretrained_models/CosyVoice2-0.5B')
# cosyvoice = CosyVoice2('../pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
prompt_speech_16k_dict = {
    '中文女': load_wav('../asset/zero_shot_prompt.wav', 16000),
    '中文男': load_wav('../asset/cross_lingual_prompt.wav', 16000),
    }
print(cosyvoice.list_available_spks())

def test_stream_one_shot_basic():
    """
    测试 stream_one_shot 基本功能
    """
    # 初始化模型
    model_dir = ' cosyvoice-tts'  # 替换为实际的模型路径
    # cosyvoice = CosyVoice(model_dir)
    
    # 准备测试数据
    tts_text = "你好，这是一个测试文本，用于验证stream_one_shot功能是否正常工作。"
    prompt_text = "今天天气真好，适合出去散步。"
    
    # 加载或生成提示语音
    # 这里创建一个简单的语音样本用于测试
    sample_rate = 22050
    duration = 3  # 3秒
    prompt_speech_16k = torch.randn(1, sample_rate * duration)
    
    # 调用 stream_one_shot
    results = []
    for i, chunk in enumerate(cosyvoice.stream_one_shot(tts_text, prompt_text, prompt_speech_16k)):
        print(f"生成第 {i} 个语音块，形状: {chunk.shape}")
        results.append(chunk)
    
    # 合并所有语音块
    if results:
        full_speech = torch.cat(results, dim=1)
        print(f"完整语音形状: {full_speech.shape}")
        # 保存结果
        torchaudio.save('test_stream_one_shot_output.wav', full_speech, sample_rate)
        print("测试结果已保存到 test_stream_one_shot_output.wav")

def test_stream_one_shot_with_stop_flag():
    """
    测试带停止标志的 stream_one_shot
    """
    # 初始化模型
    model_dir = 'cosyvoice-tts'  # 替换为实际的模型路径
    # cosyvoice = CosyVoice(model_dir)
    
    # 准备测试数据
    tts_text = "这是一段较长的测试文本，用来验证停止标志是否能正常工作。这段文本应该会被分割成多个部分进行处理。"
    prompt_text = "我们正在测试语音合成系统。"
    
    # 创建语音样本
    sample_rate = 22050
    duration = 3
    prompt_speech_16k = torch.randn(1, sample_rate * duration)
    
    # 创建停止标志
    stop_flag = threading.Event()
    
    # 启动一个线程在2秒后设置停止标志
    def set_stop_flag():
        time.sleep(2)
        stop_flag.set()
        print("停止标志已设置")
    
    stop_thread = threading.Thread(target=set_stop_flag)
    stop_thread.start()
    
    # 调用 stream_one_shot
    results = []
    start_time = time.time()
    for i, chunk in enumerate(cosyvoice.stream_one_shot(tts_text, prompt_text, prompt_speech_16k, stop_generation_flag=stop_flag)):
        elapsed = time.time() - start_time
        print(f"[{elapsed:.2f}s] 生成第 {i} 个语音块，形状: {chunk.shape}")
        results.append(chunk)
    
    stop_thread.join()
    
    # 检查是否提前停止
    print(f"总共生成了 {len(results)} 个语音块")
    if results:
        full_speech = torch.cat(results, dim=1)
        torchaudio.save('test_stream_one_shot_stopped.wav', full_speech, sample_rate)
        print("提前停止测试结果已保存到 test_stream_one_shot_stopped.wav")

def test_stream_one_shot_with_generator():
    """
    测试使用生成器作为输入文本的 stream_one_shot
    """
    # 初始化模型
    # model_dir = 'cosyvoice-tts'  # 替换为实际的模型路径
    # cosyvoice = CosyVoice(model_dir)
    
    # 定义文本生成器
    def text_generator():
        yield '收到好友从远方寄来的生日礼物，'
        yield '那份意外的惊喜与深深的祝福'
        yield '让我心中充满了甜蜜的快乐，'
        yield '笑容如花儿般绽放。'
    
    prompt_text = "希望你以后能够做的比我还好呦。"
    
    # 创建语音样本
    sample_rate = 22050
    duration = 3
    prompt_speech_16k = torch.randn(1, sample_rate * duration)
    
    # 调用 stream_one_shot
    results = []
    for i, chunk in enumerate(cosyvoice.stream_one_shot(text_generator(), prompt_text, prompt_speech_16k)):
        print(f"生成第 {i} 个语音块，形状: {chunk.shape}")
        results.append(chunk)
    
    # 合并所有语音块
    if results:
        full_speech = torch.cat(results, dim=1)
        torchaudio.save('test_stream_one_shot_generator.wav', full_speech, sample_rate)
        print("生成器输入测试结果已保存到 test_stream_one_shot_generator.wav")

if __name__ == "__main__":
    print("开始测试 stream_one_shot 功能...")
    
    print("\n=== 测试基本功能 ===")
    test_stream_one_shot_basic()
    
    print("\n=== 测试带停止标志 ===")
    test_stream_one_shot_with_stop_flag()
    
    print("\n=== 测试生成器输入 ===")
    test_stream_one_shot_with_generator()
    
    print("\n所有测试完成!")