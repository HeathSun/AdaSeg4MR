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
import time
import re
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import threading
segmentation_model = YOLO("yolo11m-seg.pt") 
segmentation_active = threading.Event()
executor = ThreadPoolExecutor(max_workers=2)

wake_word = 'Ada'
groq_client = Groq(api_key="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")
genai.configure(api_key="AIzaSyC58SeZxg_0fhw5AZfXfmMT1-uB3EVOrww")
openai_client = OpenAI(api_key="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")
web_cam = cv2.VideoCapture(0)
# Add a global variable to control voice interaction
use_voice_interaction = False  # Set this to False to disable voice interaction

sys_msg = (
    'You are a multi-modal AI voice assistant named Ada, after the British computer scientist Ada Lovelace,'
    ' Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed'
    'text prompt that will be attached to their transcribed voice prompt, Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response beforeadding new tokens to the response. '
    'Do not expect or request images, just use the context if added.Use all of the context of this conversation so your response is relevant to the conversation.'
    ' Make your responses clear and concise, avoiding any verbosity.'
)

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
        'respond with only one selection from this list: ["extract clipboard", "real-time segmentation", "take screenshot", "capture webcam", "None"] \n'
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
    
def start_segmentation():
    print("Starting segmentation...")
    speak("I am starting segmentation, please wait for the results.")
    segmentation_active.set()

    def segmentation_task():
        while segmentation_active.is_set():
            results = segmentation_model.predict(0, save=True, show=True, conf=0.15)
            if results:
                first_result = results[0]
                object_counts = {}
                for obj in first_result.boxes.cls:
                    obj_name = segmentation_model.names[int(obj)]
                    object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                response = "I have detected the following instances: " + \
                           ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])
                print(response)
                speak(response)
            else:
                print("No objects detected in the current frame.")
                speak("No objects detected in the current frame.")
            time.sleep(1)  # Adjust the sleep time as needed

    executor.submit(segmentation_task)


    
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
            start_segmentation()
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
            
        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        print(f'Ada: {response}')
        speak(response)

    
def start_listening():
    if use_voice_interaction:
        with source as s:
            print('Listening...')
            r.adjust_for_ambient_noise(s, duration = 2)
        print('\nSay', wake_word, 'followed with your prompt. \n')
        r.listen_in_background(source, callback)
        
        while True:
            time.sleep(.5)
    else:
        # If voice interaction is disabled, use command line input for testing.
        while True:
            user_input = input("User: ")
            clean_prompt = extract_prompt(user_input, wake_word)
            
            if clean_prompt:
                print(f'USER: {clean_prompt}')
                call = function_call(clean_prompt)
                
                if 'take screenshot' in call:
                    print('Taking screenshot...')
                    take_screenshot()
                    visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.png')
                elif 'real-time segmentation' in call:
                    start_segmentation()
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
