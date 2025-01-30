from groq import Groq
from PIL import ImageGrab, Image
from openai import OpenAI
import google.generativeai as genai
import pyperclip
import cv2
import pyaudio

groq_client = Groq(api_key="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")
genai.configure(api_key="AIzaSyC58SeZxg_0fhw5AZfXfmMT1-uB3EVOrww")
openai_client = OpenAI(api_key="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")
web_cam = cv2.VideoCapture(0)

sys_msg = (
    'You are a multi-modal AI voice assistant, Your user may or may not have attached a photo for context '
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
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
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
        model = 'tts-1'
        voice = 'nova'
        response_format = 'pcm'
        input = text
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size = 1024):
            if stream_start:
                player_stream.write(chunk)
            elif chunk > silence_threshold:
                player_stream.write(chunk)
                stream_start = True

    
    
while True:
    prompt = input('USER: ')
    call = function_call(prompt)
    
    if 'take screenshot' in call:
        print('Taking screenshot...')
        take_screenshot()
        visual_context = vision_prompt(prompt=prompt, photo_path='screenshot.png')
    elif 'capture webcam' in call:
        print('Capturing webcam...')
        web_cam_capture()
        visual_context = vision_prompt(prompt=prompt, photo_path='webcam.png')
    elif 'extract clipboard' in call:
        print('Copying clipboard text...')
        paste = get_clipboard()
        prompt = f'{prompt}\n\n CLIPBOARD CONTENT: {paste}'
        visual_context = None
    else:
        visual_context = None
        
    response = groq_prompt(prompt=prompt, img_context=visual_context)
    print(f'AI: {response}')
    speak(response)