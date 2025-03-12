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
import re
import json
from ultralytics import YOLO
import numpy as np
from collections import deque
segmentation_model = YOLO("yolo11m-seg.pt") 

wake_word = 'Lady Ada'
# 在文件开头的全局变量区域添加摄像头选择
CAMERA_SOURCE = 0  # 0: 内置摄像头, 1: 外接webcam, 2: DroidCam

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
ENABLE_RECORDING = False  # 设置为False可以禁用视频和音频的录制

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

# 修改function_call函数，整合新的意图检测
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
        return "real-time segmentation"
    
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
        start_segmentation(prompt)
    except Exception as e:
        # 简化错误消息，避免大量重复输出
        if "object of type 'NoneType' has no len()" in str(e):
            # 静默处理缺少检测到的物体的常见情况
            pass
        else:
            print(f"Segmentation error: {e}")

def parse_target_classes(prompt):
    """Extract target class IDs from user prompt"""
    target_classes = []
    prompt = prompt.lower()
    
    # Check each class name in mapping
    for class_name, class_id in CLASSES.items():
        if class_name in prompt:
            target_classes.append(class_id)
            
    return target_classes if target_classes else None
    
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
    os.makedirs('./runs/recordings', exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_video_path = f'./runs/recordings/recording_{timestamp}.mp4'
    
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
    
def start_segmentation(prompt):
    global overlay_frame, current_results
    
    # 如果已经在进行分割，先停止当前的分割
    if not stop_event.is_set():
        stop_event.set()
        time.sleep(0.1)  # 给一点时间让之前的线程停止
        stop_event.clear()
    
    target_classes = parse_target_classes(prompt) if prompt else None
    
    if target_classes is None:
        speak("I am starting segmentation, please wait for the results.")
    else:
        class_names = get_class_names(target_classes)
        if len(class_names) == 1:
            speak(f"I found the {class_names[0]} for you.")
        else:
            targets_str = ", ".join(class_names[:-1]) + f" and {class_names[-1]}"
            speak(f"I found {targets_str} for you.")

    while not stop_event.is_set():
        try:
            if frame_buffer is None:
                continue
                
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
                continue
            
            # 更新当前结果
            current_results = results[0] if len(results) > 0 else None
            
            # 创建透明遮罩层
            overlay = np.zeros_like(frame_buffer, dtype=np.uint8)
            
            for r in results:
                if len(r.masks) == 0:  # 如果没有检测到目标，跳过
                    continue
                    
                # 获取原始图像尺寸
                height, width = frame_buffer.shape[:2]
                
                # 处理每个检测结果
                for mask, box, cls in zip(r.masks.data, r.boxes.data, r.boxes.cls):
                    # 获取类别ID和对应的固定颜色
                    class_id = int(cls)
                    color = CLASS_COLORS[class_id]
                    class_name = [name for name, id in CLASSES.items() if id == class_id][0]
                    
                    # 直接使用YOLO的分割mask，保持原始精度
                    mask = mask.cpu().numpy()  # [H, W]
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                    mask_bool = mask > 0.5
                    
                    # 直接应用不透明的颜色遮罩
                    overlay[mask_bool] = color
                    
                    # 绘制边界框
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加类别标签
                    label = f"{class_name}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        overlay,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        overlay,
                        label,
                        (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
            
            overlay_frame = overlay
                    
        except Exception as e:
            print(f"Segmentation error: {e}")
            continue

def stop_segmentation():
    stop_event.set()  # Signal thread to stop
    speak("Stopping segmentation")
    # 清除最后一帧的分割结果
    global overlay_frame
    overlay_frame = None

def get_instance_count(class_name):
    """获取当前场景中特定类别的实例数量"""
    if current_results is None:
        return 0
        
    class_id = CLASSES.get(class_name.lower())
    if class_id is None:
        return 0
        
    # 统计指定类别的实例数量
    count = sum(1 for cls in current_results.boxes.cls if int(cls) == class_id)
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
    """获取特定类别物体的位置描述"""
    global overlay_frame  # 需要修改overlay_frame来添加箭头
    
    if current_results is None:
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
    
    for box, cls in zip(current_results.boxes.data, current_results.boxes.cls):
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
            
            # 添加对新功能的处理
            if 'describe frame' in call:
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

start_listening()
