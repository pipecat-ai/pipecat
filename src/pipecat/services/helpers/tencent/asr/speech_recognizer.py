# -*- coding: utf-8 -*-
import asyncio
import sys
import hmac
import hashlib
import base64
import time
import json
import threading
from loguru import logger
import websocket
import uuid
import urllib

import websockets


def is_python3():
    if sys.version > '3':
        return True
    return False


# 实时识别语音使用
class SpeechRecognitionListener():
    '''
    reponse:
    on_recognition_start的返回只有voice_id字段。
    on_fail 只有voice_id、code、message字段。
    on_recognition_complete没有result字段。
    其余消息包含所有字段。
    字段名	类型
    code	Integer
    message	String
    voice_id	String
    message_id	String
    result	Result
    final	Integer

    Result的结构体格式为:
    slice_type	Integer
    index	Integer
    start_time	Integer
    end_time	Integer
    voice_text_str	String
    word_size	Integer
    word_list	Word Array

    Word的类型为:
    word    String
    start_time Integer
    end_time Integer
    stable_flag：Integer
    '''

    async def on_recognition_start(self, response):
        pass

    async def on_sentence_begin(self, response):
        pass

    async def on_recognition_result_change(self, response):
        pass

    async def on_sentence_end(self, response):
        pass

    async def on_recognition_complete(self, response):
        pass

    async def on_fail(self, response):
        pass


NOTOPEN = 0
STARTED = 1
OPENED = 2
FINAL = 3
ERROR = 4
CLOSED = 5

# 实时识别语音使用


class SpeechRecognizer:

    def __init__(self, appid, credential, engine_model_type, listener):
        self.result = ""
        self.credential = credential
        self.appid = appid
        self.engine_model_type = engine_model_type
        self.status = NOTOPEN
        self.ws = None
        self.wst = None
        self.voice_id = ""
        self.new_start = 0
        self.listener = listener
        self.filter_dirty = 0
        self.filter_modal = 0
        self.filter_punc = 0
        self.convert_num_mode = 0
        self.word_info = 0
        self.need_vad = 0
        self.vad_silence_time = 0
        self.hotword_id = ""
        self.hotword_list = ""
        self.reinforce_hotword = 0
        self.noise_threshold = 0
        self.voice_format = 4
        self.nonce = ""

    def set_filter_dirty(self, filter_dirty):
        self.filter_dirty = filter_dirty

    def set_filter_modal(self, filter_modal):
        self.filter_modal = filter_modal

    def set_filter_punc(self, filter_punc):
        self.filter_punc = filter_punc

    def set_convert_num_mode(self, convert_num_mode):
        self.convert_num_mode = convert_num_mode

    def set_word_info(self, word_info):
        self.word_info = word_info

    def set_need_vad(self, need_vad):
        self.need_vad = need_vad

    def set_vad_silence_time(self, vad_silence_time):
        self.vad_silence_time = vad_silence_time

    def set_hotword_id(self, hotword_id):
        self.hotword_id = hotword_id

    def set_hotword_list(self, hotword_list):
        self.hotword_list = hotword_list

    def set_voice_format(self, voice_format):
        self.voice_format = voice_format

    def set_nonce(self, nonce):
        self.nonce = nonce

    def set_reinforce_hotword(self, reinforce_hotword):
        self.reinforce_hotword = reinforce_hotword

    def set_noise_threshold(self, noise_threshold):
        self.noise_threshold = noise_threshold

    def format_sign_string(self, param):
        signstr = "asr.cloud.tencent.com/asr/v2/"
        for t in param:
            if 'appid' in t:
                signstr += str(t[1])
                break
        signstr += "?"
        for x in param:
            tmp = x
            if 'appid' in x:
                continue
            for t in tmp:
                signstr += str(t)
                signstr += "="
            signstr = signstr[:-1]
            signstr += "&"
        signstr = signstr[:-1]
        return signstr

    def create_query_string(self, param):
        signstr = "wss://asr.cloud.tencent.com/asr/v2/"
        for t in param:
            if 'appid' in t:
                signstr += str(t[1])
                break
        signstr += "?"
        for x in param:
            tmp = x
            if 'appid' in x:
                continue
            for t in tmp:
                signstr += str(t)
                signstr += "="
            signstr = signstr[:-1]
            signstr += "&"
        signstr = signstr[:-1]
        return signstr

    def sign(self, signstr, secret_key):
        hmacstr = hmac.new(secret_key.encode('utf-8'),
                           signstr.encode('utf-8'), hashlib.sha1).digest()
        s = base64.b64encode(hmacstr)
        s = s.decode('utf-8')
        return s

    def create_query_arr(self):
        query_arr = dict()

        query_arr['appid'] = self.appid
        query_arr['sub_service_type'] = 1
        query_arr['engine_model_type'] = self.engine_model_type
        query_arr['filter_dirty'] = self.filter_dirty
        query_arr['filter_modal'] = self.filter_modal
        query_arr['filter_punc'] = self.filter_punc
        query_arr['needvad'] = self.need_vad
        query_arr['convert_num_mode'] = self.convert_num_mode
        query_arr['word_info'] = self.word_info
        if self.vad_silence_time != 0:
            query_arr['vad_silence_time'] = self.vad_silence_time
        if self.hotword_id != "":
            query_arr['hotword_id'] = self.hotword_id
        if self.hotword_list != "":
            query_arr['hotword_list'] = self.hotword_list

        query_arr['secretid'] = self.credential.secret_id
        query_arr['voice_format'] = self.voice_format
        query_arr['voice_id'] = self.voice_id
        query_arr['timestamp'] = str(int(time.time()))
        if self.nonce != "":
            query_arr['nonce'] = self.nonce
        else:
            query_arr['nonce'] = query_arr['timestamp']
        query_arr['expired'] = int(time.time()) + 24 * 60 * 60
        query_arr['reinforce_hotword'] = self.reinforce_hotword
        query_arr['noise_threshold'] = self.noise_threshold
        return query_arr

    async def stop(self):
        if self.status == OPENED:
            msg = {}
            msg['type'] = "end"
            text_str = json.dumps(msg)
            self.ws.sock.send(text_str)
        if self.ws:
            if self.wst and self.wst.is_alive():
                self.wst.join()
        self.ws.close()

    async def write(self, data):

        if not self.ws.open:
            logger.error("WebSocket is closed, unable to send data")
            return

        try:
            await self.ws.send(data)
        except websockets.exceptions.ConnectionClosedOK as e:
            logger.error(f"WebSocket connection closed: {e}")
            # 处理重连逻辑或退出
        except Exception as e:
            logger.error(f"Failed to send data over WebSocket: {e}")

    async def connect_websocket(self, requrl):
        try:
            ws = await websockets.connect(requrl)
            self.ws = ws
            self.status = OPENED
            response = {'voice_id': self.voice_id}
            await self.listener.on_recognition_start(response)
            logger.info(f"{self.voice_id} recognition start")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise e

    async def start(self):
        query_arr = self.create_query_arr()
        if self.voice_id == "":
            query_arr['voice_id'] = str(uuid.uuid1())
            self.voice_id = query_arr['voice_id']
        query = sorted(query_arr.items(), key=lambda d: d[0])
        signstr = self.format_sign_string(query)

        autho = self.sign(signstr, self.credential.secret_key)
        requrl = self.create_query_string(query)
        if is_python3():
            autho = urllib.parse.quote(autho)
        else:
            autho = urllib.quote(autho)
        requrl += "&signature=%s" % autho

        try:
            # 建立 WebSocket 连接，并且创建接收消息的任务
            await self.connect_websocket(requrl)

            # 启动接收消息的任务
            asyncio.create_task(self.listen_for_messages(self.ws))

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

    async def listen_for_messages(self, ws):
        # 处理消息的循环
        try:
            if ws.open:
                async for message in ws:
                    await self.on_message(message)
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

    async def on_message(self, message):

        response = json.loads(message)
        response['voice_id'] = self.voice_id
        if response['code'] != 0:
            logger.error(f"{self.voice_id} server recognition fail {response['message']}")
            await self.listener.on_fail(response)
            return

        if "final" in response and response["final"] == 1:
            self.status = FINAL
            self.result = message
            await self.listener.on_recognition_complete(response)
            logger.info(f"{self.voice_id} recognition complete")
            return

        if "result" in response.keys():
            if response["result"]['slice_type'] == 0:
                await self.listener.on_sentence_begin(response)
            elif response["result"]["slice_type"] == 2:
                await self.listener.on_sentence_end(response)
            elif response["result"]["slice_type"] == 1:
                await self.listener.on_recognition_result_change(response)
