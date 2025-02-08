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
segmentation_model = YOLO("yolo11m-seg.pt") 

wake_word = 'Ada'
def load_config():
    with open('config.json') as f:
        return json.load(f)

config = load_config()
groq_client = Groq(api_key=config['groq_api_key'])
genai.configure(api_key=config['google_api_key'])
openai_client = OpenAI(api_key=config['openai_api_key'])
web_cam = cv2.VideoCapture(1)
# Add a global variable to control voice interaction
use_voice_interaction = False  # Set this to False to disable voice interaction
stop_event = threading.Event()

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
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    
    return response.content
    
def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt, The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "real-time segmentation", "well done","take screenshot", "capture webcam", "None"] \n'
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
    
def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Could not open webcam')
        exit()
    
    path = 'webcam_capture.png'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)
    
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
    
def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    
    with openai_client.audio.speech.with_streaming_response.create(
        model = 'tts-1',
        voice = 'sage',
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
    global stop_event
    stop_event.clear()  # Reset event
    start_segmentation(prompt)
    
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
    
def start_segmentation(prompt):
    print("Starting segmentation...")
    
    # Get target classes if prompt provided
    target_classes = parse_target_classes(prompt) if prompt else None
    
    if target_classes is None:
        speak("I am starting segmentation, please wait for the results.")
    else:
        class_names = get_class_names(target_classes)
        if len(class_names) == 1:
            speak(f"I am starting segmentation to find the {class_names[0]} for you.")
        else:
            targets_str = ", ".join(class_names[:-1]) + f" and {class_names[-1]}"
            speak(f"I am starting segmentation to find {targets_str} for you.")


    while not stop_event.is_set():
        try:
            results = segmentation_model.predict(
                source=0,
                save=False,
                show=True,
                verbose=False,
                conf=0.15,
                classes=target_classes,  # Filter specific classes
                stream=True
            )
            for r in results:
                if stop_event.is_set():
                    break
                cv2.waitKey(1)
        except Exception as e:
            print(f"Segmentation error: {e}")
            break
            
    cv2.destroyAllWindows()
    print("Segmentation stopped")

def stop_segmentation():
    stop_event.set()  # Signal thread to stop
    speak("Stopping segmentation")
        
def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
        
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)
    
    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)
        
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
        elif 'capture webcam' in call:
            print('Capturing webcam...')
            web_cam_capture()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam_capture.png')
        elif 'extract clipboard' in call:
            print('Copying clipboard text...')
            paste = get_clipboard()
            clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
            visual_context = None
        else:
            visual_context = None
        
        if ('real-time segmentation' not in call) and ('well done' not in call):
            response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
            print(f'Ada: {response}')
            speak(response)

    
# Modify start_listening to use threading for concurrency
def start_listening():
    if use_voice_interaction:
        with source as s:
            print('Listening...')
            r.adjust_for_ambient_noise(s, duration=2)
        print('\nSay', wake_word, 'followed with your prompt. \n')
        r.listen_in_background(source, callback)
        
        while True:
            time.sleep(.5)
    else:
        # If voice interaction is disabled, use command line input for testing.
        while True:
            user_input = input("Command: ")
            clean_prompt = extract_prompt(user_input, wake_word)
            
            if clean_prompt:
                print(f'User: {clean_prompt}')
                call = function_call(clean_prompt)
                
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
                elif 'capture webcam' in call:
                    print('Capturing webcam...')
                    web_cam_capture()
                    visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam_capture.png')
                elif 'extract clipboard' in call:
                    print('Copying clipboard text...')
                    paste = get_clipboard()
                    clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
                    visual_context = None
                else:
                    visual_context = None
                
                if ('real-time segmentation' not in call) and ('well done' not in call):
                    response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
                    print(f'Ada: {response}')
                    speak(response)
            else:
                print("No valid prompt detected.")

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    
    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None 
        
start_listening()
