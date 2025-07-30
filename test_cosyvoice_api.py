import requests
import json
import base64
import time
import threading

# API基础URL
BASE_URL = "http://localhost:8080"

def test_stream_sft():
    """
    测试 stream_sft 接口
    """
    print("Testing /inference/stream_sft...")
    
    payload = {
        "query": "你好，这是一个测试语音合成。",
        "speaker": "中文女",
        "speed": 1.0,
        "isStream": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/inference/stream_sft", 
                               json=payload, 
                               stream=True)
        
        if response.status_code == 200:
            request_id = response.headers.get('X-Request-ID')
            print(f"Request ID: {request_id}")
            
            # 保存音频数据
            audio_data = b""
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    audio_data += chunk
            
            # 保存为PCM文件
            with open("output_sft.pcm", "wb") as f:
                f.write(audio_data)
            print("Saved audio to output_sft.pcm")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

def test_stream_sft_json(payload, complete_callback=None):
    """
    测试 stream_sft_json 接口
    """
    print("Testing /inference/stream_sft_json...")
    # {"query":"讲个笑话","speaker":"中文女","speed":"1","isStream":"False"}
    # payload = {"query": "讲个笑话", "speaker": "中文女", "speed": "1", "isStream": "True"}
    
    try:
        response = requests.post(f"{BASE_URL}/inference/stream_sft_json", 
                               json=payload, 
                               stream=True)
        
        if response.status_code == 200:
            request_id = response.headers.get('X-Request-ID')
            print(f"Request ID: {request_id}")
            
            # 处理SSE流
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        json_str = decoded_line[5:]  # 移除 'data:' 前缀
                        try:
                            data = json.loads(json_str)
                            print(f"Received data chunk: {len(data.get('data', ''))} bytes")
                            
                        except json.JSONDecodeError:
                            print(f"Failed to decode JSON: {json_str}")
            print("Stream processing completed")
            # 处理完成回调
            if complete_callback:
                complete_callback()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

def test_stream_sft_json1():
    """
    测试 stream_sft_json1 接口
    """
    print("Testing /inference/stream_sft_json1...")
    
    payload = {
        "query": "欢迎使用CosyVoice语音合成服务。",
        "speaker": "中文女",
        "speed": 1.0,
        "isStream": "true"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/inference/stream_sft_json1", 
                               json=payload, 
                               stream=True)
        
        if response.status_code == 200:
            request_id = response.headers.get('X-Request-ID')
            print(f"Request ID: {request_id}")
            
            # 处理SSE流
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        json_str = decoded_line[5:]  # 移除 'data:' 前缀
                        try:
                            data = json.loads(json_str)
                            print(f"Received data chunk: {len(data.get('data', ''))} bytes")
                        except json.JSONDecodeError:
                            print(f"Failed to decode JSON: {json_str}")
            print("Stream processing completed")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

def test_streamclone():
    """
    测试 streamclone 接口
    """
    print("Testing /inference/streamclone...")
    
    # 注意：这个测试需要一个实际的音频文件路径
    payload = {
        "query": "这是克隆语音的测试。",
        "prompt_text": "希望你以后能够做的比我还好呦。",
        "prompt_speech": "./asset/zero_shot_prompt.wav"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/inference/streamclone", 
                               json=payload, 
                               stream=True)
        
        if response.status_code == 200:
            request_id = response.headers.get('X-Request-ID')
            print(f"Request ID: {request_id}")
            
            # 处理SSE流
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        json_str = decoded_line[5:]  # 移除 'data:' 前缀
                        try:
                            data = json.loads(json_str)
                            print(f"Received data chunk: {len(data.get('data', ''))} bytes")
                        except json.JSONDecodeError:
                            print(f"Failed to decode JSON: {json_str}")
            print("Stream processing completed")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

def test_stop_generation(request_ids):
    """
    测试停止生成接口
    """
    print("Testing /inference/stop_generation...")
    
    payload = {
        "request_ids": request_ids
    }
    
    try:
        response = requests.post(f"{BASE_URL}/inference/stop_generation", 
                               json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Stop generation result: {result}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

def test_list_speakers():
    """
    获取可用的说话人列表
    """
    print("Getting available speakers...")
    
    # 这需要在服务端添加一个新接口来获取说话人列表
    # 暂时我们只是打印一个消息
    print("Available speakers: 中文女, 中文男, 英文女, 英文男 (示例)")

if __name__ == "__main__":
    print("CosyVoice API 测试脚本")
    print("=" * 30)
    
    # # 测试各个接口
    # test_list_speakers()
    # print()
    
    # test_stream_sft()
    # print()
    
    # time.sleep(1)  # 等待一下再发送下一个请求
    

    ## 使用双流时，你可以使用生成器作为输入，这在使用文本llm模型作为输入时很有用#注意，你仍然应该有一些基本的句子分割逻辑，因为llm不能处理任意的句子长度
    # def text_generator():
    #     yield '收到好友从远方寄来的生日礼物，'
    #     yield '那份意外的惊喜与深深的祝福'
    #     yield '让我心中充满了甜蜜的快乐，'
    #     yield '笑容如花儿般绽放。'
    # for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):

    #     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # 长文本 和双流速度测试
    long_text = """毫无疑问，大多数同志对学习党的十八大精神的重要性、必要性有着比较深刻的认识，并一定能“好好学习，认真领会”。但是，学习宣传的目的，必须是为了贯彻落实。因此，中央政治局会议对学习宣传贯彻党的十八大精神进行专题部署时，要求“认真学习宣传和全面贯彻落实党的十八大精神”；坚持学以致用、用以促学，把党的十八大精神四个“落实到”；各级党委要“着力抓好落实”。在党的十八大闭幕后不久，中央政治局第一次会议便迅速对学习宣传贯彻党的十八大精神研究部署，并特别强调了“落实”，体现了新一届中央领导集体高度的政治清醒和责任自觉，为全党全国上下学习宣传贯彻党的十八大精神指明了方向。

毛泽东同志指出，“读书是学习，使用也是学习，而且是更重要的学习。”学习的目的全在于运用。只有坚持学以致用、用以促学、学用相长，把学习党的十八大报告和党章同研究解决本地区本部门经济社会发展中的重大问题结合起来，同研究解决影响人民幸福的利益问题结合起来，同研究解决党的建设中存在的突出问题结合起来，才能真正把学习党的十八大精神的收获转化为领导科学发展的实际本领、工作思路和良好作风。然而，反观时下，有些地方，少数同志，在学和用的问题上不能很好地统一起来，平时常常借口工作忙、事务繁，不学习或很少学习，即便学习也是蜻蜓点水、浮光掠影，浅尝辄止、一知半解。在他们看来，文件翻过、课程听过，就算完成“任务”了。对学习的目的不明确、不清楚，与实际脱节，不能做到学以致用、用以促学、学用相长。这种学习，是一种形而上学，是必须要克服和杜绝的。    在《唐顿庄园》中，唐顿在游戏中使用“唐顿庄园”作为游戏名称，这是在游戏中使用“唐顿庄园”作为游戏名称的例子。
    """
    #打印时间
    start_time = time.time()
    # 长文本 测试
    payload = {"query": long_text, "speaker": "中文女", "speed": "1", "isStream": "True"}
    test_stream_sft_json(payload)
    end_time = time.time()
    print("长文本用时：", end_time - start_time)
    
    
    # 双流测试
    start_time = time.time()
    # 分词器分词
    import jieba
    text_list = jieba.lcut(long_text)
    
    
    
    print(text_list)
    payload = {"query": long_text, "speaker": "中文女", "speed": "1", "isStream": "True"}
    test_stream_sft_json(payload)
    
    print()
    
    time.sleep(1)
    
    
    # test_stream_sft_json1()
    # print()
    
    # 注意：streamclone 需要有效的音频文件，所以可能需要根据实际情况调整
    # test_streamclone()
    
    print("所有测试完成")