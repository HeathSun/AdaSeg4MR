import os
import time
import random
import re
from openai import OpenAI
import json
import tempfile
import sys
import subprocess
from datetime import datetime

# 加载配置
def load_config():
    with open('config.json') as f:
        return json.load(f)

# 初始化
config = load_config()
openai_client = OpenAI(api_key=config['openai_api_key'])

def sanitize_filename(text):
    """将文本转换为安全的文件名"""
    # 移除非法字符
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    # 取前5个单词或最多30个字符
    words = text.split()
    short_text = " ".join(words[:5])
    if len(short_text) > 30:
        short_text = short_text[:30]
    return short_text

def speak(text):
    """使用OpenAI的TTS API朗读文本并保存音频文件"""
    if not text:
        return
    
    print(f"Speaking: {text}")
    
    try:
        # 创建临时文件用于存储音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()
        
        # 使用OpenAI的TTS API生成语音并直接保存到文件
        with openai_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=text
        ) as response:
            response.stream_to_file(temp_file.name)
        
        # 替换pygame播放代码
        if sys.platform == "win32":
            os.startfile(temp_file.name)
        elif sys.platform == "darwin":
            subprocess.call(["afplay", temp_file.name])
        else:
            subprocess.call(["xdg-open", temp_file.name])
        
        # 简单等待估计的播放时间 (每个字符约0.1秒)
        time.sleep(0.1 * len(text))
        
        # 创建以用户输入开头的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_num = random.randint(1000, 9999)
        safe_text = sanitize_filename(text)
        filename = f"{safe_text}_{random_num}.mp3"
        
        # 确保音频目录存在
        os.makedirs("../speak_simulation", exist_ok=True)
        
        # 保存音频文件到永久位置
        output_path = os.path.join("../speak_simulation", filename)
        with open(temp_file.name, 'rb') as temp_audio:
            with open(output_path, 'wb') as output_audio:
                output_audio.write(temp_audio.read())
        
        print(f"Audio saved to: {output_path}")
        
        # 清理临时文件
        os.unlink(temp_file.name)
        
    except Exception as e:
        print(f"Error generating or playing speech: {e}")

def main():
    """主函数，循环接收用户输入并朗读"""
    print("Welcome to Speak Simulator!")
    print("Enter text to have it spoken, or 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter text to speak: ")
        
        if user_input.lower() == 'quit':
            print("Exiting Speak Simulator. Goodbye!")
            break
        
        if not user_input.strip():
            print("Please enter some text.")
            continue
        
        # 朗读文本并保存音频
        speak(user_input)

if __name__ == "__main__":
    main()
