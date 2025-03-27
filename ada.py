from groq import Groq
from PIL import ImageGrab, Image
from openai import OpenAI
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import pyperclip
import cv2
import pyaudio
import os
import threading
import time
import keyboard
import re
import json
from ultralytics import YOLO
import numpy as np
from collections import deque
segmentation_model = YOLO("yolo11m-seg.pt") 

wake_word = 'Lady Ada'
# 在文件开头的全局变量区域添加摄像头选择
CAMERA_SOURCE = 1  # 0: 内置摄像头, 1: 外接webcam, 2: DroidCam

def load_config():
    with open('config.json') as f:
        return json.load(f)

config = load_config()
groq_client = Groq(api_key=config['groq_api_key'])
genai.configure(api_key=config['google_api_key'])
openai_client = OpenAI(api_key=config['openai_api_key'])
web_cam = cv2.VideoCapture(CAMERA_SOURCE)
web_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
web_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Add a global variable to control voice interaction
use_voice_interaction = False # Set this to False to disable voice interaction
stop_event = threading.Event()

# 在全局变量区域添加以下内容
# 用于存储对话历史的队列
chat_history = deque(maxlen=5)  # 存储最近5条对话
frame_buffer = None
overlay_frame = None

# 在全局变量区域添加
should_exit = False

# 添加全局变量用于存储当前的分割结果
current_results = None

# 在全局变量区域添加
video_writer = None
recording_started = False

# 在全局变量区域添加音频录制相关变量
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
audio_frames = []
p_record = pyaudio.PyAudio()
recording_stream = None

# 在全局变量区域添加
current_video_path = None

# 在全局变量区域添加
is_speaking = False  # 用于跟踪空格键状态

# 在全局变量区域添加
waiting_sound_thread = None
stop_waiting_sound = threading.Event()

# 在全局变量区域添加录制控制标志
ENABLE_RECORDING = True  # 设置为False可以禁用视频和音频的录制

# 在全局变量区域添加
continuous_segmentation = False  # 控制是否持续分割

# 在全局变量区域添加
detection_buffer = deque(maxlen=5)  # 存储最近5帧的检测结果

sys_msg = (
    'You are a multi-modal AI voice assistant named Ada, after the British computer scientist Ada Lovelace,'
    ' Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed'
    'text prompt that will be attached to their transcribed voice prompt, Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response beforeadding new tokens to the response. '
    'Do not expect or request images, just use the context if added.Use all of the context of this conversation so your response is relevant to the conversation.'
    ' Make your responses clear and concise, avoiding any verbosity.'
)

CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'bench': 13,
    'bird': 14,
    'cat': 15,
    'dog': 16,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    'backpack': 24,
    'umbrella': 25,
    'handbag': 26,
    'tie': 27,
    'suitcase': 28,
    'frisbee': 29,
    'skis': 30,
    'snowboard': 31,
    'sports ball': 32,
    'kite': 33,
    'baseball bat': 34,
    'baseball glove': 35,
    'skateboard': 36,
    'surfboard': 37,
    'tennis racket': 38,
    'bottle': 39,
    'wine glass': 40,
    'cup': 41,
    'fork': 42,
    'knife': 43,
    'spoon': 44,
    'bowl': 45,
    'banana': 46,
    'apple': 47,
    'sandwich': 48,
    'orange': 49,
    'broccoli': 50,
    'carrot': 51,
    'hot dog': 52,
    'pizza': 53,
    'donut': 54,
    'cake': 55,
    'chair': 56,
    'couch': 57,
    'potted plant': 58,
    'bed': 59,
    'dining table': 60,
    'toilet': 61,
    'tv': 62,
    'laptop': 63,
    'mouse': 64,
    'remote': 65,
    'keyboard': 66,
    'cell phone': 67,
    'microwave': 68,
    'oven': 69,
    'toaster': 70,
    'sink': 71,
    'refrigerator': 72,
    'book': 73,
    'clock': 74,
    'vase': 75,
    'scissors': 76,
    'teddy bear': 77,
    'hair drier': 78,
    'toothbrush': 79
}
# 在全局变量区域修改颜色映射，使用更多独特的颜色
CLASS_COLORS = {
    class_id: color for class_id, color in enumerate([
        (255, 0, 0),     # 红
        (0, 255, 0),     # 绿
        (0, 0, 255),     # 蓝
        (255, 255, 0),   # 黄
        (255, 0, 255),   # 洋红
        (0, 255, 255),   # 青
        (128, 0, 0),     # 深红
        (0, 128, 0),     # 深绿
        (0, 0, 128),     # 深蓝
        (128, 128, 0),   # 橄榄
        (128, 0, 128),   # 紫
        (0, 128, 128),   # 青绿
        (255, 128, 0),   # 橙
        (255, 0, 128),   # 粉红
        (128, 255, 0),   # 黄绿
        (0, 255, 128),   # 春绿
    ] * ((len(CLASSES) // 16) + 1))  # 重复颜色列表直到覆盖所有类别
}
convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048,
}

safety_settings = [
    {
    'category': 'HARM_CATEGORY_HARASSMENT',
    'threshold': 'BLOCK_NONE',
    },
    {
    'category': 'HARM_CATEGORY_HATE_SPEECH',
    'threshold': 'BLOCK_NONE',
    },
    {
    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'threshold': 'BLOCK_NONE',
    },
    {
    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
    'threshold': 'BLOCK_NONE',
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              safety_settings=safety_settings,
                              generation_config=generation_config)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device = 'cpu',
    compute_type = 'int8',
    cpu_threads = num_cores // 2,
    num_workers = num_cores // 2,
)

r = sr.Recognizer()
source = sr.Microphone()

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama-3.1-8b-instant')
    response = chat_completion.choices[0].message
    convo.append(response)
    
    return response.content
    
# 添加意图匹配函数
def detect_intent(prompt):
    """使用Groq API来检测用户意图"""
    sys_msg = ('''
You are an intent classification AI. Analyze the user's input and determine which of the following intents it most closely matches:
1. find_objects - User wants to find or locate specific objects in their view
2. describe_scene - User wants a description of what's currently visible
3. count_objects - User wants to count specific objects
4. position_query - User wants to know where specific objects are located
5. take_screenshot - User wants to capture a screenshot
6. clipboard_extract - User wants to extract text from clipboard
7. quit_request - User wants to exit the program
8. general_chat - None of the above, user just wants a normal conversation
9. real-time segmentation - User wants to segment all objects in real-time
10. visual_question - User is asking a specific question about what they can see
Respond ONLY with the intent name (e.g., "find_objects", "general_chat", etc.) without any explanation.
    ''')
    
    intent_convo = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=intent_convo, 
            model='llama-3.1-8b-instant',
            temperature=0.1,  # 低温度以获得更确定的结果
            max_tokens=10     # 只需要简短的答案
        )
        intent = chat_completion.choices[0].message.content.strip().lower()
        print(f"Detected intent: {intent}")
        return intent
    except Exception as e:
        print(f"Error detecting intent: {e}")
        return "general_chat"  # 默认为一般对话

# 修改translate_user_classes函数，优化多类别处理
def translate_user_classes(prompt):
    """
    将用户输入的非标准类别名称转译为YOLO支持的标准类别
    例如："men and women" -> "person", "laptop computer" -> "laptop"
    支持多类别识别，如："find people and cars" -> ["person", "car"]
    """
    # 构建系统提示
    classes_list = ", ".join(CLASSES.keys())
    sys_prompt = f"""
You are a computer vision assistant. Your task is to map user's natural language descriptions 
to the standard YOLO object detection classes.

Available YOLO classes: {classes_list}

Rules:
1. Map user terms to the closest matching YOLO class(es)
2. Consider synonyms, plurals, and related terms
3. Return ONLY the matching class name(s) without any explanation
4. If multiple classes match, return them separated by commas
5. If no classes match, return "NONE"
6. Be comprehensive - identify ALL relevant classes mentioned in the query

Examples:
- "men and women" -> "person"
- "laptop computer" -> "laptop"
- "dining table with food" -> "dining table"
- "coca cola bottle" -> "bottle"
- "mobile phone" -> "cell phone"
- "golden retriever" -> "dog"
- "unicorn" -> "NONE"
- "find people and cars" -> "person, car"
- "look for cats and dogs" -> "cat, dog"
"""

    user_prompt = f"User query: {prompt}\nMatching YOLO classes:"
    
    # 调用Groq API进行分析
    try:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,  # 低温度以获得更确定的结果
            max_tokens=50
        )
        
        # 解析响应
        mapped_classes = response.choices[0].message.content.strip()
        
        # 如果没有匹配的类别
        if mapped_classes.upper() == "NONE":
            return None
        
        # 分割多个类别
        class_list = [cls.strip() for cls in mapped_classes.split(',')]
        
        # 验证返回的类别是否在CLASSES中
        valid_classes = []
        for cls in class_list:
            if cls in CLASSES:
                valid_classes.append(cls)
        
        return valid_classes if valid_classes else None
    
    except Exception as e:
        print(f"Error translating user classes: {e}")
        return None

# 修改parse_target_classes函数，集成类别转译功能
def parse_target_classes(prompt):
    """Extract target class IDs from user prompt with translation support"""
    # 先尝试直接匹配
    target_classes = []
    prompt = prompt.lower()
    
    # 检查每个类名是否在提示中
    for class_name, class_id in CLASSES.items():
        if class_name in prompt:
            target_classes.append(class_id)
    
    # 如果没有直接匹配到，尝试使用转译
    if not target_classes:
        translated_classes = translate_user_classes(prompt)
        if translated_classes:
            for class_name in translated_classes:
                class_id = CLASSES.get(class_name)
                if class_id is not None:
                    target_classes.append(class_id)
    
    # 如果找到了类别，则返回，否则返回None
    return target_classes if target_classes else None

# 修改function_call函数中的实时分割部分，增加对未支持类别的处理
def function_call(prompt):
    # 保持原有的直接匹配逻辑
    # 检查是否是描述框架请求
    description_keywords = ['describe frame', 'what do you see', 'analyze scene', 'describe scene']
    prompt_lower = prompt.lower()
    
    # 如果包含描述关键词
    if any(keyword in prompt_lower for keyword in description_keywords):
        return "describe frame"
    
    # 添加位置查询的检测
    location_keywords = ['where', 'location', 'position']
    
    # 检查是否是查找请求
    if 'find' in prompt_lower and 'for me' in prompt_lower:
        # 检查是否请求查找不支持的对象
        translated_classes = translate_user_classes(prompt)
        if translated_classes is None:
            # 这是一个不支持分割的对象，返回visual question
            return "visual question"
        return "real-time segmentation"
    
    # 其余代码保持不变
    if any(keyword in prompt_lower for keyword in location_keywords):
        for class_name in CLASSES.keys():
            if class_name in prompt_lower:
                return "position query"
    
    # 添加一些关键词来检测数量查询
    counting_keywords = ['how many', 'count', 'number of']
    
    # 检查是否是数量查询
    if any(keyword in prompt_lower for keyword in counting_keywords):
        for class_name in CLASSES.keys():
            if class_name in prompt_lower:
                return "count objects"
    
    # 使用AI意图检测
    intent = detect_intent(prompt)
    
    # 根据检测到的意图返回对应的功能
    if intent == "find_objects":
        # 检查是否请求查找不支持的对象
        translated_classes = translate_user_classes(prompt)
        if translated_classes is None:
            # 这是一个不支持分割的对象，返回visual question
            return "visual question"
        return "real-time segmentation"
    elif intent == "describe_scene":
        return "describe frame"
    elif intent == "visual_question":
        return "visual question"
    elif intent == "count_objects":
        return "count objects"
    elif intent == "position_query":
        return "position query"
    elif intent == "take_screenshot":
        return "take screenshot"
    elif intent == "clipboard_extract":
        return "extract clipboard"
    elif intent == "quit_request":
        return "quit"
    else:  # general_chat或其他任何情况
        # 回退到原有的功能调用逻辑
        sys_msg = (
            'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
            'taking a screenshot, calling no functions is best for a voice assistant to respond '
            'to the users prompt, The webcam can be assumed to be a normal laptop webcam facing the user. You will '
            'respond with only one selection from this list: ["extract clipboard", "real-time segmentation", "well done",'
            '"take screenshot", "count objects", "position query", "quit", "None"] \n'
            'Do not respond with anything but the most logical selection from that list with no explanations. Format the'
            'function call name exactly as I listed.'
        )
        
        function_convo = [{'role': 'system', 'content': sys_msg},
                          {'role': 'user', 'content': prompt}]
                          
        chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
        response = chat_completion.choices[0].message
        
        return response.content

def take_screenshot():
    path = 'screenshot.png'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality = 15)
    
def get_clipboard():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        return 'Error: Could not get clipboard content'
    
def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis Al that provides semtantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the Al assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the Al'
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text
    
def play_waiting_sound():
    """播放等待音效"""
    try:
        import winsound
        winsound.PlaySound('./detector.wav', winsound.SND_ASYNC | winsound.SND_LOOP)
        
        while not stop_waiting_sound.is_set():
            time.sleep(0.1)
        
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as e:
        print(f"Error playing waiting sound: {e}")

def start_waiting_sound():
    """开始播放等待音效"""
    try:
        global waiting_sound_thread, stop_waiting_sound
        stop_waiting_sound.clear()
        waiting_sound_thread = threading.Thread(target=play_waiting_sound)
        waiting_sound_thread.daemon = True
        waiting_sound_thread.start()
    except Exception as e:
        print(f"Error starting waiting sound: {e}")

def stop_waiting_sound_thread():
    """停止等待音效"""
    try:
        global stop_waiting_sound
        if stop_waiting_sound is not None:
            stop_waiting_sound.set()
            time.sleep(0.2)  # 给一点时间让音效停止
    except Exception as e:
        print(f"Error stopping waiting sound: {e}")
    
def speak(text):
    """修改speak函数，根据voice_interaction标志决定是否生成语音"""
    chat_history.append(f"Ada: {text}")  # 添加到历史记录
    
    # 停止等待音效
    stop_waiting_sound_thread()
    
    # 如果禁用了语音交互，则只显示文本不生成语音
    if not use_voice_interaction:
        return
    
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    
    with openai_client.audio.speech.with_streaming_response.create(
        model = 'tts-1',
        voice = 'shimmer',
        response_format = 'pcm',
        input = text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                amplitude = max(chunk)
                if amplitude > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text
    
# Function to handle segmentation in a separate thread
def segmentation_thread(prompt):
    """在单独的线程中执行分割"""
    try:
        result = start_segmentation(prompt)
        if result is None or (hasattr(result, 'boxes') and len(result.boxes) == 0):
            print("No objects were detected during segmentation.")
            speak("I couldn't find any of the objects you were looking for.")
    except Exception as e:
        # 简化错误消息，避免大量重复输出
        if "object of type 'NoneType' has no len()" in str(e):
            # 静默处理缺少检测到的物体的常见情况
            print("No objects detected in this frame.")
        else:
            print(f"Segmentation error: {e}")
    
def get_class_names(target_classes):
    """Convert class IDs to names"""
    names = []
    for class_id in target_classes:
        for name, id in CLASSES.items():
            if id == class_id:
                names.append(name)
                break
    return names
    
def add_text_to_frame(frame, text_lines):
    """在帧的左下角添加文本，区分用户和Ada的发言"""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    padding = 10
    line_spacing = 30
    
    # 计算起始y坐标（从底部往上）
    y = height - padding - (len(text_lines) - 1) * line_spacing
    
    for line in text_lines:
        # 添加黑色背景
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(frame, 
                     (padding, int(y - text_height)), 
                     (padding + text_width, int(y + 5)), 
                     (0, 0, 0), 
                     -1)
        
        # 根据说话者选择不同的文本颜色
        if line.startswith("User:"):
            text_color = (255, 200, 0)  # 用户发言用金黄色
        else:
            text_color = (255, 255, 255)  # Ada发言用白色
            
        cv2.putText(frame, line, 
                    (padding, int(y)), 
                    font, 
                    font_scale, 
                    text_color, 
                    thickness)
        y += line_spacing
    return frame

def setup_video_writer():
    """设置视频写入器"""
    global video_writer, current_video_path
    
    # 如果禁用录制，直接返回
    if not ENABLE_RECORDING:
        return None
        
    # 确保recordings目录存在
    os.makedirs('../runs/recordings', exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_video_path = f'../runs/recordings/recording_{timestamp}.mp4'
    
    # 获取视频参数
    width = int(web_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(web_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0
    
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(current_video_path, fourcc, fps, (width, height))
    return current_video_path

def display_video_thread():
    """持续显示视频流的线程函数"""
    global frame_buffer, overlay_frame, should_exit, recording_started, video_writer
    
    cv2.namedWindow('Ada Video Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ada Video Feed', 1280, 720)
    
    # 初始化视频写入器
    if not recording_started and ENABLE_RECORDING:
        output_path = setup_video_writer()
        print(f"Recording to: {output_path}")
        recording_started = True
    elif not ENABLE_RECORDING:
        print("Video recording is disabled.")
    
    while not should_exit:
        ret, frame = web_cam.read()
        if not ret:
            continue
            
        frame_buffer = frame.copy()
        display_frame = frame.copy()
        
        # 如果有分割结果，叠加显示
        if overlay_frame is not None:
            display_frame = cv2.addWeighted(display_frame, 0.7, overlay_frame, 0.3, 0)
            
        # 添加对话历史
        display_frame = add_text_to_frame(display_frame, list(chat_history))
        
        # 写入帧到视频文件
        if video_writer is not None and ENABLE_RECORDING:
            video_writer.write(display_frame)
        
        cv2.imshow('Ada Video Feed', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理资源
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    web_cam.release()
    
# 修改process_segmentation_results函数，确保正确处理多类别掩码
def process_segmentation_results(results, frame):
    """将分割结果可视化到叠加层，支持多类别同时显示"""
    if results is None:
        return None
        
    # 创建透明叠加层
    overlay = np.zeros_like(frame)
    
    height, width = frame.shape[:2]
    
    # 处理每个检测到的物体
    try:
        # 处理分割掩码（如果有）- 先处理掩码再处理边界框，避免边界框被掩码覆盖
        if hasattr(results, 'masks') and results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                class_id = int(results.boxes.cls[i])
                color = CLASS_COLORS[class_id]
                
                # 转换掩码为numpy数组并调整大小
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_LINEAR)
                mask_np = mask_np > 0.5  # 二值化
                
                # 应用掩码并保持不同类别的颜色区分
                alpha = 0.4  # 降低透明度，使多个类别可以区分
                overlay[mask_np] = (overlay[mask_np] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        
        # 处理边界框
        for i, box in enumerate(results.boxes.xyxy):
            # 获取类别ID和名称
            class_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])  # 获取置信度
            
            # 获取对应的类别名称
            class_name = None
            for name, id in CLASSES.items():
                if id == class_id:
                    class_name = name
                    break
                    
            if class_name is None:
                continue
                
            # 设置颜色
            color = CLASS_COLORS[class_id]
            
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box)
            
            # 绘制边界框
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # 添加置信度文本
            conf_text = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # 计算文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                conf_text, font, font_scale, thickness
            )
            
            # 文本位置（在框的上方）
            text_x = x1
            text_y = y1 - 5
            
            # 确保文本在图像内
            if text_y < text_height:
                text_y = y1 + text_height + 5
                
            # 绘制文本背景
            cv2.rectangle(
                overlay, 
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                color, 
                -1
            )
            
            # 绘制文本
            cv2.putText(
                overlay,
                conf_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
    
    except Exception as e:
        print(f"Error in processing segmentation results: {e}")
    
    return overlay

# 添加函数来获取最佳检测结果（物体最多的那一帧）
def get_best_detection():
    """从缓冲区中获取物体检测最多的结果"""
    global detection_buffer
    
    if not detection_buffer:
        return None
        
    # 找出包含最多物体的结果
    max_objects = 0
    best_result = None
    
    for result in detection_buffer:
        if result is not None and hasattr(result, 'boxes'):
            num_objects = len(result.boxes)
            if num_objects > max_objects:
                max_objects = num_objects
                best_result = result
                
    return best_result

# 修改start_segmentation函数，优化多类别处理
def start_segmentation(prompt):
    """开始分割对象，支持多类别检测"""
    global segmentation_model, segmentation_mode, current_results, stop_event, overlay_frame
    global continuous_segmentation, detection_buffer
    
    # 清空检测缓冲区
    detection_buffer.clear()
    
    # 重置停止事件
    stop_event.clear()
    
    # 启用连续分割模式
    continuous_segmentation = True

    # 解析目标类别
    target_classes = parse_target_classes(prompt)
    
    # 如果没有指定目标类别，使用所有类别
    if target_classes is None:
        print(f"No specific object class detected in prompt. Showing all detectable objects.")
        target_classes = list(CLASSES.values())
    else:
        print(f"Target classes: {target_classes}")
        
    # 获取目标类别名称
    target_names = get_class_names(target_classes)
    print(f"Looking for: {', '.join(target_names)}")
    speak(f"Looking for {', '.join(target_names)}")
    
    segmentation_mode = True
    segmentation_start_time = time.time()
    first_detection = False
    
    try:
        # 连续处理帧，直到收到停止信号
        while not stop_event.is_set():
            if frame_buffer is None:
                time.sleep(0.1)
                continue
                
            # 处理当前帧
            try:
                # 添加超时检查，长时间找不到则停止
                if time.time() - segmentation_start_time > 30 and not first_detection:
                    print("Segmentation timeout - no objects found within 30 seconds.")
                    speak("I couldn't find the objects you're looking for after searching for 30 seconds.")
                    break
                    
                # 预测结果
                results = segmentation_model.predict(
                    source=frame_buffer,
                    save=False,
                    show=False,
                    verbose=False,
                    conf=0.15,
                    classes=target_classes,
                    retina_masks=True
                )
                
                # 添加结果检查
                if results is None or len(results) == 0:
                    # 处理没有结果的情况
                    overlay_frame = None  # 清除叠加层
                    detection_buffer.append(None)  # 将空结果添加到缓冲区
                    continue
                
                # 检查是否找到了任何物体
                if len(results[0].boxes) == 0:
                    # 没有找到物体，清除叠加层并继续下一帧
                    overlay_frame = None
                    detection_buffer.append(None)  # 将空结果添加到缓冲区
                    continue
                
                # 更新当前结果
                current_results = results[0] if len(results) > 0 else None
                
                # 将当前结果添加到缓冲区
                if current_results is not None:
                    detection_buffer.append(current_results)
                
                # 处理分割结果并更新叠加层
                overlay_frame = process_segmentation_results(current_results, frame_buffer)
                
                # 如果找到物体，通知用户（仅第一次）
                if current_results is not None and len(current_results.boxes) > 0 and not first_detection:
                    first_detection = True
                    
                    # 统计检测到的每种类别的数量
                    found_classes = {}
                    for cls in current_results.boxes.cls:
                        class_id = int(cls)
                        class_name = [name for name, id in CLASSES.items() if id == class_id][0]
                        found_classes[class_name] = found_classes.get(class_name, 0) + 1
                    
                    # 构建检测结果消息
                    detection_message = "I found "
                    class_details = []
                    for class_name, count in found_classes.items():
                        if count == 1:
                            class_details.append(f"a {class_name}")
                        else:
                            class_details.append(f"{count} {class_name}s")
                    
                    detection_message += ", ".join(class_details)
                    
                    print(f"Detection result: {detection_message}")
                    speak(detection_message)
                
                # 如果不是连续模式且已找到物体，则退出循环
                if not continuous_segmentation and first_detection:
                    break
                
            except Exception as e:
                # 只在非NoneType错误时打印错误信息
                if "object of type 'NoneType' has no len()" not in str(e):
                    print(f"Segmentation processing error: {e}")
                # 继续处理下一帧
                continue
                
            # 短暂休眠，减少CPU使用率
            time.sleep(0.05)
            
    except Exception as e:
        # 捕获其他异常
        print(f"Segmentation error: {e}")
    finally:
        # 结束分割模式
        segmentation_mode = False
        if not continuous_segmentation:
            overlay_frame = None  # 如果不是连续模式，清除叠加层
        return current_results

# 修改stop_segmentation函数
def stop_segmentation():
    """停止分割并清除显示结果"""
    global stop_event, overlay_frame, continuous_segmentation
    
    stop_event.set()  # Signal thread to stop
    continuous_segmentation = False  # 禁用连续分割模式
    overlay_frame = None  # 清除叠加层
    speak("Stopping segmentation")

def get_instance_count(class_name):
    """获取当前场景中特定类别的实例数量，使用多帧缓冲技术"""
    # 获取最佳检测结果（物体最多的那一帧）
    best_result = get_best_detection()
    
    if best_result is None:
        return 0
        
    class_id = CLASSES.get(class_name.lower())
    if class_id is None:
        return 0
        
    # 统计指定类别的实例数量
    count = sum(1 for cls in best_result.boxes.cls if int(cls) == class_id)
    return count

def format_count_response(count, class_name):
    """根据数量格式化响应文本"""
    if count == 1:
        return f"I can see 1 {class_name} in the current scene."
    else:
        return f"I can see {count} {class_name}s in the current scene."

# 在callback和start_listening中修改数量查询的处理
def handle_count_query(clean_prompt):
    """处理数量查询的统一函数，支持多类别查询"""
    prompt_lower = clean_prompt.lower()
    found_classes = []
    responses = []
    
    # 收集所有在提示中提到的类别
    for class_name in CLASSES.keys():
        if class_name in prompt_lower:
            found_classes.append(class_name)
    
    if not found_classes:
        return False
        
    # 获取每个类别的数量并生成响应
    for class_name in found_classes:
        count = get_instance_count(class_name)
        if count == 1:
            responses.append(f"1 {class_name}")
        else:
            responses.append(f"{count} {class_name}s")
    
    # 组合所有响应
    if len(responses) == 1:
        final_response = f"I can see {responses[0]} in the current scene."
    elif len(responses) == 2:
        final_response = f"I can see {responses[0]} and {responses[1]} in the current scene."
    else:
        final_response = "I can see " + ", ".join(responses[:-1]) + f", and {responses[-1]} in the current scene."
    
    print(f'Ada: {final_response}')
    speak(final_response)
    return True

def get_region(x, y, width, height):
    """确定坐标在屏幕的哪个区域"""
    x_ratio = x / width
    y_ratio = y / height
    
    # 水平位置
    if x_ratio < 0.33:
        h_pos = "left"
    elif x_ratio > 0.67:
        h_pos = "right"
    else:
        h_pos = "center"
        
    # 垂直位置
    if y_ratio < 0.33:
        v_pos = "upper"
    elif y_ratio > 0.67:
        v_pos = "lower"
    else:
        v_pos = "middle"
        
    # 组合位置
    if h_pos == "center" and v_pos == "middle":
        return "center"
    elif h_pos == "center":
        return f"{v_pos} {h_pos}"
    elif v_pos == "middle":
        return h_pos
    else:
        return f"{v_pos} {h_pos}"

def draw_arrow(frame, target_center, color=(255, 255, 255), thickness=3):
    """在帧上绘制指向目标的箭头"""
    height, width = frame.shape[:2]
    # 箭头起点（屏幕中心稍微偏上的位置）
    start_point = (width // 2, int(height * 0.4))
    
    # 计算箭头方向
    dx = target_center[0] - start_point[0]
    dy = target_center[1] - start_point[1]
    length = (dx**2 + dy**2)**0.5
    
    if length == 0:
        return frame
    
    # 箭头终点就是目标中心点
    end_point = target_center
    
    # 绘制箭头
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.3)
    return frame

def get_position_descriptor(index, total_count):
    """根据索引和总数返回位置描述词"""
    if total_count == 1:
        return ""
    elif total_count == 2:
        return "left" if index == 0 else "right"
    else:
        if index == 0:
            return "leftmost"
        elif index == total_count - 1:
            return "rightmost"
        elif total_count % 2 == 1 and index == total_count // 2:
            return "middle"
        else:
            # 判断物体在画面左半边还是右半边
            if index < total_count // 2:
                # 在左半边,从左数第几个
                from_left = index + 1
                if from_left == 2:
                    return "second from the left"
                elif from_left == 3:
                    return "third from the left" 
                else:
                    return f"{from_left}th from the left"
            else:
                # 在右半边,从右数第几个
                from_right = total_count - index
                if from_right == 2:
                    return "second from the right"
                elif from_right == 3:
                    return "third from the right"
                else:
                    return f"{from_right}th from the right"

def get_relative_position(target_box, other_boxes, other_classes):
    """找到最近的其他物体并描述相对位置"""
    tx1, ty1, tx2, ty2 = map(int, target_box[:4])
    target_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
    
    # 找到左边和右边最近的物体
    nearest_left = None
    nearest_right = None
    left_dist = float('inf')
    right_dist = float('inf')
    left_class = None 
    right_class = None
    
    for box, cls in zip(other_boxes, other_classes):
        x1, y1, x2, y2 = map(int, box[:4])
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # 计算水平距离
        dist = abs(center[0] - target_center[0])
        
        # 如果在目标左边
        if center[0] < target_center[0]:
            if dist < left_dist:
                left_dist = dist
                nearest_left = center
                left_class = cls
        # 如果在目标右边
        elif center[0] > target_center[0]:
            if dist < right_dist:
                right_dist = dist
                nearest_right = center
                right_class = cls
                
    # 如果左右都没有物体
    if nearest_left is None and nearest_right is None:
        return ""
        
    # 如果只有左边有物体
    if nearest_right is None:
        left_class_name = [name for name, id in CLASSES.items() if id == int(left_class)][0]
        return f" at the right of the {left_class_name}"
        
    # 如果只有右边有物体
    if nearest_left is None:
        right_class_name = [name for name, id in CLASSES.items() if id == int(right_class)][0]
        return f" at the left of the {right_class_name}"
        
    # 如果两边都有物体
    left_class_name = [name for name, id in CLASSES.items() if id == int(left_class)][0]
    right_class_name = [name for name, id in CLASSES.items() if id == int(right_class)][0]
    
    if left_class_name == right_class_name:
        return f" between two {left_class_name}s"
    else:
        return f" between the {left_class_name} and the {right_class_name}"

def get_object_positions(class_name):
    """获取特定类别物体的位置描述，使用多帧缓冲技术"""
    global overlay_frame  # 需要修改overlay_frame来添加箭头
    
    # 获取最佳检测结果（物体最多的那一帧）
    best_result = get_best_detection()
    
    if best_result is None:
        return "I don't see any objects currently being segmented."
        
    class_id = CLASSES.get(class_name.lower())
    if class_id is None:
        return f"I don't recognize the object type '{class_name}'."
        
    # 获取所有该类别的边界框和其他类别的边界框
    target_boxes = []
    target_centers = []
    other_boxes = []
    other_classes = []
    
    height, width = frame_buffer.shape[:2]
    
    for box, cls in zip(best_result.boxes.data, best_result.boxes.cls):
        if int(cls) == class_id:
            target_boxes.append(box)
            x1, y1, x2, y2 = map(int, box[:4])
            target_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        else:
            other_boxes.append(box)
            other_classes.append(cls)
    
    if not target_boxes:
        return f"I don't see any {class_name} in the current scene."
        
    # 如果只有一个目标物体
    if len(target_boxes) == 1:
        x1, y1, x2, y2 = map(int, target_boxes[0][:4])
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        region = get_region(center[0], center[1], width, height)
        relative = get_relative_position(target_boxes[0], other_boxes, other_classes)
        
        # 添加箭头到overlay
        if overlay_frame is not None:
            overlay_frame = draw_arrow(overlay_frame, (int(center[0]), int(center[1])))
        
        return f"The {class_name} is in the {region} of the screen{relative}."
        
    # 如果有多个目标物体
    positions = []
    # 根据x坐标排序
    sorted_indices = sorted(range(len(target_centers)), 
                          key=lambda i: target_centers[i][0])
    
    # 为每个物体添加箭头
    if overlay_frame is not None:
        for idx in sorted_indices:
            center = target_centers[idx]
            overlay_frame = draw_arrow(overlay_frame, (int(center[0]), int(center[1])))
    
    for i, idx in enumerate(sorted_indices):
        box = target_boxes[idx]
        center_x, center_y = target_centers[idx]
        
        # 使用新的位置描述函数
        position_word = get_position_descriptor(i, len(target_boxes))
        if position_word:  # 如果有位置描述词
            region = get_region(center_x, center_y, width, height)
            relative = get_relative_position(box, other_boxes, other_classes)
            positions.append(f"The {position_word} {class_name} is in the {region} of the screen{relative}")
    
    return " ".join(positions)

def handle_position_query(clean_prompt):
    """处理位置查询"""
    prompt_lower = clean_prompt.lower()
    
    # 检查是否是位置查询
    location_keywords = ['where', 'location', 'position']
    if not any(keyword in prompt_lower for keyword in location_keywords):
        return False
        
    # 查找提到的类别
    for class_name in CLASSES.keys():
        if class_name in prompt_lower:
            response = get_object_positions(class_name)
            print(f'Ada: {response}')
            speak(response)
            return True
            
    return False

# 在callback和start_listening中添加位置查询的处理
def callback(recognizer, audio):
    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
            
        prompt_text = wav_to_text(prompt_audio_path)
        clean_prompt = extract_prompt(prompt_text, wake_word)
        
        if clean_prompt:
            chat_history.append(f"User: {clean_prompt}")
            print(f'USER: {clean_prompt}')
            call = function_call(clean_prompt)
            
            # 处理不同的功能调用
            if 'position query' in call:
                if handle_position_query(clean_prompt):
                    return
            elif 'count objects' in call:
                if handle_count_query(clean_prompt):
                    return
            elif 'take screenshot' in call:
                print('Taking screenshot...')
                take_screenshot()
                visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.png')
            elif 'real-time segmentation' in call:
                segmentation_thread_instance = threading.Thread(target=segmentation_thread, kwargs={'prompt': clean_prompt})
                segmentation_thread_instance.start()
                visual_context = None
            elif 'well done' in call:
                print('Thank you...')
                stop_segmentation()
                visual_context = None
            elif 'extract clipboard' in call:
                print('Copying clipboard text...')
                paste = get_clipboard()
                clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
                visual_context = None
            elif 'quit' in call.lower():
                quit_program()
            else:
                visual_context = None
            
            # 处理需要LLM响应的情况
            if ('real-time segmentation' not in call) and ('well done' not in call):
                response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
                print(f'Ada: {response}')
                speak(response)
    except Exception as e:
        print(f"Error in voice recognition: {e}")

def on_press(event):
    """处理按键按下事件"""
    global is_speaking, source
    try:
        # 修改空格键检测方式
        if event.name == 'space' or (  # 首先检查name属性
            hasattr(event, 'vk') and event.vk == 32) or (  # 然后检查虚拟键码
            isinstance(event, keyboard.KeyCode) and event.char == ' '):  # 最后检查字符
            
            if not is_speaking:
                # 开始录音
                is_speaking = True
                print("Recording started... Press space again to stop.")
                try:
                    with source as s:
                        audio = r.listen(s, timeout=None, phrase_time_limit=None)
                        # 处理录音
                        handle_audio(audio)
                except Exception as e:
                    print(f"Error recording: {e}")
                finally:
                    is_speaking = False
                    print("Recording stopped.")
            else:
                # 如果正在录音，这次按空格会在 r.listen 中触发 stop_listening
                r.stop_listening()
    except Exception as e:
        print(f"Error in key press handler: {e}")

def on_release(event):
    """处理按键释放事件"""
    pass  # 不需要处理释放事件

def start_listening():
    # 启动视频显示线程
    video_thread = threading.Thread(target=display_video_thread)
    video_thread.daemon = True
    video_thread.start()
    
    # 启动音频录制
    setup_audio_recording()
    
    if use_voice_interaction:
        from pynput import keyboard
        
        # 设置键盘监听
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        with source as s:
            print('Listening...')
            r.adjust_for_ambient_noise(s, duration=1)
            r.energy_threshold = 300
            r.dynamic_energy_threshold = True
            r.dynamic_energy_adjustment_damping = 0.15
            r.dynamic_energy_ratio = 1.5
            r.pause_threshold = 0.5
            r.operation_timeout = None
            r.phrase_threshold = 0.3
            r.non_speaking_duration = 0.3
        
        print('\nPress space to start/stop recording.\n')  # 更新提示信息
        
        while True:
            time.sleep(.5)
    else:
        # 修改命令行输入处理部分
        while True:
            user_input = input("Command: ")
            clean_prompt = extract_prompt(user_input, wake_word)
            
            if clean_prompt:
                chat_history.append(f"User: {clean_prompt}")
                print(f'User: {clean_prompt}')
                
                call = function_call(clean_prompt)
                
                # 添加对describe frame的处理
                if 'describe frame' in call:
                    describe_frame()
                    continue
                elif 'position query' in call:
                    if handle_position_query(clean_prompt):
                        continue
                elif 'count objects' in call:
                    if handle_count_query(clean_prompt):
                        continue
                elif 'quit' in call.lower():
                    quit_program()
                    break
                
                if 'take screenshot' in call:
                    print('Taking screenshot...')
                    take_screenshot()
                    visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.png')
                elif 'real-time segmentation' in call:
                    # Start segmentation in a new thread to avoid blocking
                    segmentation_thread_instance = threading.Thread(target=segmentation_thread, kwargs={'prompt': clean_prompt})
                    segmentation_thread_instance.start()
                    visual_context = None
                elif 'well done' in call:
                    print('Thank you...')
                    stop_segmentation()
                    visual_context = None
                elif 'extract clipboard' in call:
                    print('Copying clipboard text...')
                    paste = get_clipboard()
                    clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
                    visual_context = None
                elif 'visual question' in call:
                    answer_visual_question(clean_prompt)
                    continue
                else:
                    visual_context = None
                
                if ('real-time segmentation' not in call) and ('well done' not in call):
                    response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
                    print(f'Ada: {response}')
                    # 添加Ada的回复到对话历史
                    chat_history.append(f"Ada: {response}")
                    speak(response)

def extract_prompt(transcribed_text, wake_word):
    """直接返回转录的文本"""
    return transcribed_text.strip() if transcribed_text else None
        
def setup_audio_recording():
    """设置音频录制"""
    global recording_stream
    
    # 如果禁用录制，不启动音频录制
    if not ENABLE_RECORDING:
        return
        
    recording_stream = p_record.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    # 启动音频录制线程
    audio_thread = threading.Thread(target=record_audio_thread)
    audio_thread.daemon = True
    audio_thread.start()

def record_audio_thread():
    """持续录制音频的线程函数"""
    global audio_frames
    
    # 如果禁用录制，直接返回
    if not ENABLE_RECORDING:
        return
        
    while not should_exit:
        try:
            data = recording_stream.read(CHUNK, exception_on_overflow=False)
            audio_frames.append(data)
        except Exception as e:
            print(f"Error recording audio: {e}")
            continue

def save_audio_recording(base_path):
    """保存音频录制文件"""
    import wave
    
    # 使用与视频相同的文件名，但是是wav格式
    audio_path = base_path.rsplit('.', 1)[0] + '_audio.wav'
    
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p_record.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
    
    print(f"Audio recording saved to: {audio_path}")

def quit_program():
    """退出程序的函数"""
    global should_exit, stop_event, video_writer, recording_stream, p_record
    
    # 停止分割（如果正在进行）
    stop_event.set()
    
    # 说再见并等待语音完成
    speak("Goodbye! Have a great day!")
    print("Exiting program...")
    time.sleep(2)  # 等待语音播放完成
    
    # 设置退出标志来关闭视频流
    should_exit = True
    
    # 给视频流一点时间完成清理
    time.sleep(0.5)
    
    # 保存视频和音频
    if video_writer is not None and ENABLE_RECORDING:
        video_writer.release()
        print(f"Video recording saved to: {current_video_path}")
        
        # 停止音频录制并保存
        if recording_stream is not None:
            recording_stream.stop_stream()
            recording_stream.close()
            save_audio_recording(current_video_path)
        p_record.terminate()
    
    # 退出程序
    os._exit(0)
        
def handle_audio(audio):
    """处理录制的音频"""
    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
            
        prompt_text = wav_to_text(prompt_audio_path)
        clean_prompt = extract_prompt(prompt_text, wake_word)
        
        if clean_prompt:
            chat_history.append(f"User: {clean_prompt}")
            print(f'USER: {clean_prompt}')
            call = function_call(clean_prompt)
            
            # 开始播放等待音效
            start_waiting_sound()
            
            # 处理实例推理请求
            if call.startswith("instance_reasoning:"):
                target_class = call.split(":", 1)[1]
                instance_reasoning(clean_prompt, target_class)
                return
            elif 'describe frame' in call:
                describe_frame()
                return
            elif 'visual question' in call:
                answer_visual_question(clean_prompt)
                return
            elif 'position query' in call:
                if handle_position_query(clean_prompt):
                    return
            elif 'count objects' in call:
                if handle_count_query(clean_prompt):
                    return
            elif 'take screenshot' in call:
                print('Taking screenshot...')
                take_screenshot()
                visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.png')
            elif 'real-time segmentation' in call:
                segmentation_thread_instance = threading.Thread(target=segmentation_thread, kwargs={'prompt': clean_prompt})
                segmentation_thread_instance.start()
                visual_context = None
            elif 'well done' in call:
                print('Thank you...')
                stop_segmentation()
                visual_context = None
            elif 'extract clipboard' in call:
                print('Copying clipboard text...')
                paste = get_clipboard()
                clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
                visual_context = None
            elif 'quit' in call.lower():
                quit_program()
        else:
            visual_context = None
        
        if ('real-time segmentation' not in call) and ('well done' not in call):
            response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
            print(f'Ada: {response}')
            speak(response)
    except Exception as e:
        print(f"Error in voice recognition: {e}")
        
# 添加以下函数来处理"describe frame"功能
def describe_frame():
    """分析并描述当前视频帧"""
    global frame_buffer
    
    if frame_buffer is None:
        speak("I can't see anything right now.")
        return
    
    # 保存当前帧
    frame_path = 'current_frame_analysis.png'
    cv2.imwrite(frame_path, frame_buffer)
    
    print("Analyzing what I can see...")
    
    try:
        # 准备API请求
        from base64 import b64encode
        
        # 读取图像并编码为base64
        with open(frame_path, "rb") as image_file:
            image_data = b64encode(image_file.read()).decode('utf-8')
        
        # 使用Groq的视觉模型API
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Describe what you see in the current frame with a very concise description in one sentence."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            max_tokens=300
        )
        
        description = response.choices[0].message.content
        print(f"Ada: {description}")
        speak(description)
        
    except Exception as e:
        error_message = f"I encountered an error analyzing the image: {str(e)}"
        print(error_message)
        speak(error_message)
    
    # 清理临时文件
    try:
        os.remove(frame_path)
    except:
        pass

# 添加新函数来回答关于视频帧的具体问题
def answer_visual_question(question):
    """分析当前视频帧并回答用户的具体问题"""
    global frame_buffer
    
    if frame_buffer is None:
        speak("I can't see anything right now to answer your question.")
        return
    
    # 保存当前帧
    frame_path = 'current_frame_question.png'
    cv2.imwrite(frame_path, frame_buffer)
    
    print(f"Analyzing frame to answer: {question}")
    
    try:
        # 准备API请求
        from base64 import b64encode
        
        # 读取图像并编码为base64
        with open(frame_path, "rb") as image_file:
            image_data = b64encode(image_file.read()).decode('utf-8')
        
        # 使用Groq的视觉模型API，包含用户的具体问题
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": f"Look at this image and answer this question very concisely: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content
        print(f"Ada: {answer}")
        speak(answer)
        
    except Exception as e:
        error_message = f"I encountered an error analyzing the image: {str(e)}"
        print(error_message)
        speak(error_message)
    
    # 清理临时文件
    try:
        os.remove(frame_path)
    except:
        pass
        
# 添加新的实例推理函数
def instance_reasoning(question, target_class):
    """分析特定类别的每个实例并提供描述"""
    global current_results, frame_buffer
    
    if current_results is None or frame_buffer is None:
        speak("I don't have any segmentation results to analyze yet.")
        return
    
    # 获取该类别在CLASSES中的ID
    target_class_id = None
    for class_name, class_id in CLASSES.items():
        if class_name.lower() == target_class.lower():
            target_class_id = class_id
            break
    
    if target_class_id is None:
        speak(f"I don't recognize {target_class} as a known object class.")
        return
    
    # 从分割结果中提取指定类别的所有框
    boxes = []
    class_ids = []
    
    # 检查是否有boxes属性
    if hasattr(current_results, 'boxes') and current_results.boxes is not None:
        for i, cls in enumerate(current_results.boxes.cls):
            if int(cls) == target_class_id:
                box = current_results.boxes.xyxy[i].cpu().numpy()
                boxes.append(box)
                class_ids.append(int(cls))
    
    if not boxes:
        speak(f"I couldn't find any {target_class} in the current segmentation results.")
        return
    
    # 根据x坐标排序（从左到右）
    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    
    instance_descriptions = []
    position_descriptors = get_position_descriptors(len(boxes))
    
    print(f"Analyzing {len(boxes)} {target_class} instances...")
    
    # 处理每个实例
    for i, idx in enumerate(sorted_indices):
        box = boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        
        # 确保边界不超出图像范围
        height, width = frame_buffer.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # 裁剪边界框
        cropped_instance = frame_buffer[y1:y2, x1:x2]
        
        if cropped_instance.size == 0:
            continue
            
        # 保存裁剪区域
        instance_path = f'instance_{i}.png'
        cv2.imwrite(instance_path, cropped_instance)
        
        # 分析单个实例
        description = analyze_instance(instance_path, target_class, position_descriptors[i])
        instance_descriptions.append(description)
        
        # 清理临时文件
        try:
            os.remove(instance_path)
        except:
            pass
    
    # 整合所有实例描述
    if len(instance_descriptions) == 1:
        final_description = f"The {target_class} {instance_descriptions[0]}"
    else:
        final_description = "; ".join(instance_descriptions)
    
    # 缓存结果到记忆文件
    cache_instance_reasoning(target_class, final_description)
    
    print(f"Ada: {final_description}")
    speak(final_description)
    return final_description

def analyze_instance(image_path, class_name, position):
    """分析单个实例并返回描述"""
    try:
        from base64 import b64encode
        
        # 读取图像并编码为base64
        with open(image_path, "rb") as image_file:
            image_data = b64encode(image_file.read()).decode('utf-8')
        
        # 使用Groq的视觉模型API分析实例
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": f"Analyze this cropped image of a {class_name}. Describe its notable visual features (color, condition, shape, etc.) in one brief phrase."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            max_tokens=50
        )
        
        features = response.choices[0].message.content.strip()
        
        # 构建描述
        return f"{position} is {features}"
        
    except Exception as e:
        print(f"Error analyzing instance: {e}")
        return f"{position} is visible"

def get_position_descriptors(count):
    """根据物体数量生成位置描述词"""
    if count == 1:
        return [""]
    elif count == 2:
        return ["on the left", "on the right"]
    elif count == 3:
        return ["on the left", "in the middle", "on the right"]
    else:
        result = []
        for i in range(count):
            if i == 0:
                result.append("on the far left")
            elif i == count - 1:
                result.append("on the far right")
            elif i == 1:
                result.append("on the left")
            elif i == count - 2:
                result.append("on the right")
            else:
                relative_position = (i / (count - 1)) * 100
                result.append(f"at position {i+1} from the left")
        return result

def cache_instance_reasoning(target_class, description):
    """缓存实例推理结果到JSON文件"""
    cache_file = "instance_memory.json"
    
    # 加载现有缓存或创建新缓存
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                memory = json.load(f)
        except:
            memory = {}
    else:
        memory = {}
    
    # 更新缓存
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    memory[timestamp] = {
        "class": target_class,
        "description": description
    }
    
    # 保存缓存
    with open(cache_file, 'w') as f:
        json.dump(memory, f, indent=2)
        
start_listening()
