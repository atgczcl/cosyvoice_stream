import os
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.utils.file_utils import logging
import threading


class CosyVoice:

    def __init__(self, model_dir, load_jit=True, load_onnx=False):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'], configs
            ['feat_extractor'], '{}/campplus.onnx'.format(model_dir),
            '{}/speech_tokenizer_v1.onnx'.format(model_dir),
            '{}/spk2info.pt'.format(model_dir), instruct, configs[
            'allowed_special'])
        self.model = CosyVoiceModel(configs['llm'], configs['flow'],
            configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir), '{}/flow.pt'.format(
            model_dir), '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.fp16.zip'.format(
                model_dir), '{}/llm.llm.fp16.zip'.format(model_dir),
                '{}/flow.encoder.fp32.zip'.format(model_dir))
        if load_onnx:
            self.model.load_onnx('{}/flow.decoder.estimator.fp32.onnx'.
                format(model_dir))
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream,
                speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(
                    speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k,
        stream=False, speed=1.0):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text,
                prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream,
                speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(
                    speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=
        False, speed=1.0):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.
                format(self.model_dir))
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_cross_lingual(i,
                prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream,
                speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(
                    speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=
        False, speed=1.0):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(
                self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False
            )
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_instruct(i, spk_id,
                instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream,
                speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(
                    speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=
        False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k,
            prompt_speech_16k)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream,
            speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / 22050
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (
                time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()

    def stream_clone(self, tts_text, prompt_text, prompt_speech_16k, stream
        =True, speed=1.0, stop_generation_flag: threading.Event=None):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            if (stop_generation_flag is not None and stop_generation_flag.
                is_set()):
                break
            model_input = self.frontend.frontend_zero_shot(i, prompt_text,
                prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream,
                speed=speed):
                if (stop_generation_flag is not None and
                    stop_generation_flag.is_set()):
                    break
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech clone len {}, rtf {}'.format(
                    speech_len, (time.time() - start_time) / speech_len))
                tts_speech_chunk = model_output['tts_speech']
                yield tts_speech_chunk
                start_time = time.time()

    def stream_sft(self, tts_text: str, spk_id: int, stream: bool=True,
        speed: float=1.0, stop_generation_flag: threading.Event=None):
        """
        流式处理文本转语音的功能。
        
        :param tts_text: 要转换的文本
        :param spk_id: 说话人ID
        :param stream: 是否流式传输
        :param speed: 语速
        :param stop_generation_flag: 停止生成标志
        """
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            if (stop_generation_flag is not None and stop_generation_flag.
                is_set()):
                break
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream,
                speed=speed):
                if (stop_generation_flag is not None and
                    stop_generation_flag.is_set()):
                    break
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech sft len {}, rtf {}'.format(
                    speech_len, (time.time() - start_time) / speech_len))
                tts_speech_chunk = model_output['tts_speech']
                yield tts_speech_chunk
                start_time = time.time()
