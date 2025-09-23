#!/home/ver/env2/bin/python

import asyncio
from gpiozero import Button
from signal import pause
from google import genai
from groq import Groq

from google.genai import types
import time
import cv2
import threading
import sounddevice as sd
import soundfile as sf
import queue
import os
import numpy as np
import wave
from flask import Flask, request
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
import subprocess
from Bluetooth_Connection import *
import base64
import re
import io
import signal
import atexit
import functools

# ----------- Google / Groq API client -------------
client = genai.Client(api_key="")
client_groq = Groq(api_key="")
vision_model = 0

# ----------- Globals -----------------------
state = 0
loop = None
image_bytes = None
recognized_text = None
recording_task = None
speech_task = None
q = queue.Queue()
searching_task = None
groq_task = None
google_task = None
voice = None

# Flask 
app_text = None
tts_speed = 1
received_text_from_app = asyncio.Event()

# Event trackers
cancel_event = asyncio.Event()
record_stop_event = asyncio.Event()
speech_task_event = asyncio.Event()

# Camera lock
camera_in_use = False
camera_lock = threading.Lock()

# Camera resolution settings
CAMERA_WIDTH = 1280   # Default: 1280 (HD)
CAMERA_HEIGHT = 720   # Default: 720 (HD)

# Audio
channels = 1
dtype = 'int16'
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"
audio_lock = threading.Lock()  # Thread lock for audio operations

# Loading sound
loading_stop_event = threading.Event()
loading_thread = None

# Shared executor used for running blocking SDK/network calls
blocking_executor = ThreadPoolExecutor(max_workers=4)
MODEL_CALL_TIMEOUT = 15  # seconds, tune down if you want faster cancel response

###############################################################
# ------------------------- Flask App ------------------------#
###############################################################
app = Flask(__name__)

def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

@app.route('/receive_mac', methods=['POST'])
def receive_mac():
    global loop
    mac = request.data.decode().strip()
    print(f"Received MAC: {mac}")
    try:
        filename = os.path.dirname(__file__)[:-1] + "saved_mac.txt"
        with open(filename, 'w') as file:
            file.write(mac)
            print("Mac written")
        bt_status = pair_trust_connect(mac)

        if bt_status:
            if loop:
                time.sleep(1)
                asyncio.run_coroutine_threadsafe(text_to_speech("Connected to device"), loop)

        return "OK", 200
    except subprocess.CalledProcessError:
        return "Failed", 500

@app.route('/receive_text', methods=['POST'])
def receive_text():
    global app_text, received_text_from_app, loop
    text = request.data.decode()
    app_text = text
    if loop:
        loop.call_soon_threadsafe(received_text_from_app.set)
        print(f"Received text: {text}")
    return "OK", 200

@app.route('/start', methods=['POST'])
def start():
    global app_text, received_text_from_app, loop
    app_text = "start"
    if loop:
        loop.call_soon_threadsafe(received_text_from_app.set)
        print(f"Received Start")
    return "OK", 200

@app.route('/stop', methods=['POST'])
def stop():
    global app_text, received_text_from_app, loop
    app_text = "stop"
    if loop:
        loop.call_soon_threadsafe(received_text_from_app.set)
        print(f"Received Stop")
    return "OK", 200

@app.route('/cancel', methods=['POST'])
def cancel():
    global app_text, received_text_from_app, loop
    app_text = "cancel"
    cancel_event.set()
    if loop:
        loop.call_soon_threadsafe(received_text_from_app.set)
        print(f"Received Cancel")
    return "OK", 200


###############################################################
# ------------------------- Audio Handling -------------------#
###############################################################
def safe_play_audio(data, samplerate, blocking=False):
    """Safely play audio with proper stream management"""
    global audio_lock
    with audio_lock:
        try:
            # Stop any currently playing audio
            sd.stop()
            time.sleep(0.1)
            # Play new audio
            sd.play(data, samplerate=samplerate, blocking=blocking)
        except Exception as e:
            print(f"Error in safe_play_audio: {e}")

def safe_stop_audio():
    """Safely stop all audio"""
    global current_audio_stream, audio_lock
    with audio_lock:
        try:
            sd.stop()
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in safe_stop_audio: {e}")

def play_sound_effect(file_path):
    try:
        data, samplerate = sf.read(file_path, dtype="int16")
        safe_play_audio(data, samplerate)
    except Exception as e:
        print(f"Error playing sound effect: {e}")

###############################################################
# ------------------------- Audio Recording-------------------#
###############################################################
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    q.put(indata.copy())

def record_audio_sync():
    device_info = sd.query_devices(None, "input")
    samplerate = int(device_info["default_samplerate"])
    frames = []

    with sd.InputStream(samplerate=samplerate, channels=channels,
                        dtype=dtype, callback=audio_callback,
                        blocksize=CHUNK):
        print("Recording... Press stop or cancel to end.")
        while not record_stop_event.is_set() and not cancel_event.is_set():
            try:
                data = q.get(timeout=0.5)
                frames.append(data)
            except queue.Empty:
                continue

    print("Recording stopped. Saving...")
    audio = np.concatenate(frames, axis=0)
    speech_file_path = os.path.dirname(__file__) + WAVE_OUTPUT_FILENAME
    
    with wave.open(speech_file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    return frames

async def record_audio():
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, record_audio_sync)

###############################################################
# ------------------------- Camera Capture -------------------#
###############################################################
def capture_image(resolution_width=None, resolution_height=None):
    global CAMERA_WIDTH, CAMERA_HEIGHT
    
    # Use global settings if not specified
    if resolution_width is None:
        resolution_width = CAMERA_WIDTH
    if resolution_height is None:
        resolution_height = CAMERA_HEIGHT
    global camera_in_use, camera_lock
    
    # Wait for camera to be available
    max_wait_time = 10  # 10 seconds max wait
    wait_time = 0
    
    while camera_in_use and wait_time < max_wait_time:
        print("Camera in use, waiting...")
        time.sleep(0.5)
        wait_time += 0.5
    
    if camera_in_use:
        print("Camera timeout - camera still in use")
        return None
    
    # Acquire camera lock
    with camera_lock:
        camera_in_use = True
        try:
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                try:
                    cam = cv2.VideoCapture(camera_index)
                    if cam.isOpened():
                        # Enable MJPEG for better resolution
                        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        
                        # Set desired resolution
                        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
                        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
                        cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

                        # Optional: wait a short time for autofocus to adjust
                        time.sleep(0.4)
                        
                        # Capture frame
                        ret, frame = cam.read()
                        if ret:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                            _, buffer = cv2.imencode('.jpg', frame)
                            cam.release()
                            return buffer.tobytes()
                        else:
                            print(f"Failed to get image from camera {camera_index}")
                    cam.release()
                except Exception as e:
                    print(f"Error with camera {camera_index}: {e}")
                    continue
            print("No working camera found")
            return None
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
        finally:
            # Always release the camera lock
            camera_in_use = False

async def wait_for_camera():
    """Wait for camera to be available asynchronously"""
    global camera_in_use
    max_wait_time = 10  # 10 seconds max wait
    wait_time = 0
    
    while camera_in_use and wait_time < max_wait_time:
        print("Camera in use, waiting...")
        await asyncio.sleep(0.5)
        wait_time += 0.5
    
    if camera_in_use:
        print("Camera timeout - camera still in use")
        return False
    
    return True

###############################################################
# ------------------ Text to speech --------------------------#
###############################################################
async def text_to_speech(text):
    global cancel_event, speech_task_event
    try:
        if cancel_event.is_set():
            return

        speech_task_event.set()
        # Generate TTS with gTTS
        tts = gTTS(text=text, lang="en")
        mp3_fp = io.BytesIO()

        if cancel_event.is_set():
            return
            
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # Decode MP3 into waveform
        data, samplerate = sf.read(mp3_fp, dtype="int16")

        # Play audio with proper stream management
        safe_play_audio(data, samplerate, blocking=False)
        
        # Calculate audio duration and wait for completion
        audio_duration = len(data) / samplerate  # Duration in seconds
        max_wait_time = audio_duration + 2  # Add 2 seconds buffer
        
        # Wait for audio to complete or be cancelled
        wait_time = 0
        while wait_time < max_wait_time:
            await asyncio.sleep(0.05)  # Check every 50ms
            if cancel_event.is_set():
                return
            wait_time += 0.05
    finally:
        speech_task_event.clear()


###############################################################
# ------------------ Loading ---------------------------------#
###############################################################
def play_loading_sound(file_path):
    global loading_thread, loading_stop_event
    
    try:
        # Stop any currently playing audio first
        safe_stop_audio()
        # Clear the stop event for new loading sound
        loading_stop_event.clear()
        data, samplerate = sf.read(file_path, dtype="int16")

        def loop_sound():
            try:
                while not loading_stop_event.is_set():
                    # Create a new stream for each iteration to avoid conflicts
                    safe_play_audio(data, samplerate, blocking=True)
                    if loading_stop_event.is_set() or cancel_event.is_set:
                        break
                    time.sleep(1.5)
            except Exception as e:
                print(f"Loading sound error: {e}")
           
        # Stop previous loading thread if it exists
        if loading_thread and loading_thread.is_alive():
            loading_stop_event.set()
            loading_thread.join(timeout=1.0)
        
        loading_thread = threading.Thread(target=loop_sound, daemon=True)
        loading_thread.start()

    except Exception as e:
        print(f"Error playing loading sound: {e}")

def stop_loading_sound():
    global loading_thread, loading_stop_event
    
    loading_stop_event.set()
    # Stop audio and wait for thread to finish
    safe_stop_audio()
    
    # Wait for loading thread to finish
    if loading_thread and loading_thread.is_alive():
        loading_thread.join(timeout=2.0)

        
###############################################################
# ------------------ Task Determination ----------------------#
###############################################################
def task_selection():
    
    input_message = f"Classify the following user input into one of these categories by providing only the corresponding number. \
                    Do not include any additional text or explanation. User Input: {recognized_text}. Categories: \
                    1. **Live Task Execution:** The user wants an action performed immediately based on a detected event or condition. (e.g., Tell me when you see a cat, Notify me if the light turns red.) \
                    2. **Image Analysis/Question:** The user is requesting a description of an image, or asking a question about its content. (e.g., Describe what's in front of me, Is there a laptop in this picture?, Whats in front of me?, Can you see a tree?) \
                    3. **General Conversation/Information Retrieval:** The user is engaging in a general conversation or asking a factual question. (e.g., What's the weather like?,Tell me a joke,Who is the prime minister?) \
                    4. **Image Reading:** The user wants something read or some information from text in the image. (e.g., Can you read this sign?, Does this have gluten?, What are the ingredients in this?, How much salt is in this?)  \
                    5. **Help:** The user needs general assistance (e.g., Help)"

    if cancel_event.is_set():
        return
    
    chat_completion = client_groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="llama-3.3-70b-versatile",)
        
    return chat_completion.choices[0].message.content

###############################################################
# ------------------------- Groq -----------------------------#
###############################################################
def _run_groq_chat_sync(model, messages, **kwargs):
    # This runs in a separate thread (blocking)
    return client_groq.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )

async def send_to_model(model, text):
    # Wait for camera to be available
    if not await wait_for_camera():
        await text_to_speech("Unable to capture image with camera, please try again")
        return None

    image_bytes_local = capture_image()
    if not image_bytes_local:
        await text_to_speech("Error capturing image")
        return None

    base_64_image = base64.b64encode(image_bytes_local).decode('utf-8')
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base_64_image}"}}
            ]
        }
    ]

    if cancel_event.is_set():
        return None

    loop = asyncio.get_running_loop()
    blocking_future = loop.run_in_executor(
        blocking_executor,
        functools.partial(
            _run_groq_chat_sync,
            model,
            messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
    )

    try:
        completion = await asyncio.wait_for(blocking_future, timeout=MODEL_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        print("Model call timed out")
        stop_loading_sound()
        await text_to_speech("Request timed out, please try again")
        return None
    except asyncio.CancelledError:
        print("Send to model coroutine cancelled")
        return None
    except Exception as e:
        print(f"Exception calling Groq model: {e}")
        stop_loading_sound()
        await text_to_speech("Error from server, please try again later")
        return None

    if cancel_event.is_set():
        return None

    try:
        response = completion.choices[0].message.content
    except Exception as e:
        print(f"Unexpected completion format: {e}")
        return None

    return response

###############################################################
# ------------------------- Gemini----------------------------#
###############################################################
def _run_gemini_generate_sync(recognized_text: str, image_bytes: bytes):
    """Blocking Gemini API call. Should only be run in executor threads."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    # Configure generation settings
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            f"Answer this question directly in less than 50 words, use the image if it's required: {recognized_text}"
        ],
        config=config,
    )



###############################################################
# ------------------------- Tasks ----------------------------#
###############################################################
## Task 1 -> Searching
async def object_searching_async(recognized_text):
    global vision_model, cancel_event, speech_task, speech_task_event

    if cancel_event.is_set():
        return

    # First Groq call (get search object)
    messages = [{
        "role": "user",
        "content": f"Tell me, if there is one, what object is being searched for in this input: {recognized_text}, only tell me the object no additional information or tell me 'no object' if there was no object given",
    }]

    loop = asyncio.get_running_loop()
    blocking_future = loop.run_in_executor(
        blocking_executor,
        functools.partial(
            _run_groq_chat_sync,
            "llama-3.3-70b-versatile",
            messages,
            temperature=1,
            max_completion_tokens=256
        )
    )

    try:
        completion = await asyncio.wait_for(blocking_future, timeout=MODEL_CALL_TIMEOUT)
    except Exception as e:
        print(f"Error getting object to search for: {e}")
        return

    searching = completion.choices[0].message.content

    if not cancel_event.is_set():
        speech_task = asyncio.create_task(text_to_speech(f"Searching for {searching}"))

    if re.search(r"no object", searching, re.IGNORECASE):
        if not cancel_event.is_set():
            await text_to_speech("Unable to find object in the speech input")
        return

    object_found = False
    # Searching loop
    while not object_found and not cancel_event.is_set():
        if vision_model:
            model_running = "meta-llama/llama-4-maverick-17b-128e-instruct"
            vision_model = 0
        else:
            model_running = "meta-llama/llama-4-scout-17b-16e-instruct"
            vision_model = 1

        if cancel_event.is_set():
            return

        image_bytes_local = capture_image()
        if not image_bytes_local or not searching.strip():
            await asyncio.sleep(1)
            continue

        base_64_image = base64.b64encode(image_bytes_local).decode('utf-8')
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Does this image contain {searching}, answer only with yes or no and if the answer is yes tell me where it is using clockface coordinates using only between 10 and 2 o'clock with 10 o'clock being leftmost and 2 o'clock being rightmost and estimate its distance from me in metres"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base_64_image}"}}
            ]
        }]

        blocking_future = loop.run_in_executor(
            blocking_executor,
            functools.partial(
                _run_groq_chat_sync,
                model_running,
                messages,
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False
            )
        )

        try:
            completion = await asyncio.wait_for(blocking_future, timeout=MODEL_CALL_TIMEOUT)
        except asyncio.TimeoutError:
            print("object_searching model call timed out; retrying")
            if not cancel_event.is_set():
                await text_to_speech("Timed out searching, trying again")
            await asyncio.sleep(1)
            continue
        except Exception as e:
            print(f"Error in object searching request: {e}")
            if not cancel_event.is_set():
                await text_to_speech("Error from server, please try again later")
            return

        if cancel_event.is_set():
            return

        response = completion.choices[0].message.content
        print(f"Groq Response: {response}")

        if re.search(r"\byes\b", response, re.IGNORECASE):
            object_found = True
            while speech_task_event.is_set():
                await asyncio.sleep(0.1)
            if not cancel_event.is_set():
                await text_to_speech(f"Found {searching}. {response}")
            return
        else:
            await asyncio.sleep(10)

## Task 2 / Task 4 -> Image description / Reading
async def image_description():
    global speech_task, vision_model, speech_task_event
        
    if vision_model:
        model_running = "meta-llama/llama-4-maverick-17b-128e-instruct"
        vision_model = 0
    else:
        model_running = "meta-llama/llama-4-scout-17b-16e-instruct"
        vision_model = 1
    
    text = f"Answer this about the given image in less than 50 words and for context I am blind but do not say anything about me being blind, give me only information relevant to the question asked and use metric units for anything, answer only the following: {recognized_text}"

    loading_stop_event.clear()

    if cancel_event.is_set():
        return

    play_loading_sound("waiting_sound.wav")
    response = await send_to_model(model_running, text)

    if response == None:
        return 
    else:
        while speech_task_event.is_set():
            await asyncio.sleep(0.1)
        speech_task = asyncio.create_task(text_to_speech(response))

## Task 3 -> Google search
async def google_searching(user_input):
    if not await wait_for_camera():
        await text_to_speech("Camera is busy, please try again later")
        return

    image_bytes = capture_image()
    if not image_bytes:
        await text_to_speech("Error capturing image")
        return

    if cancel_event.is_set():
        return

    loop = asyncio.get_running_loop()
    play_loading_sound("waiting_sound.wav")

    text = f"Answer this question directly in less than 50 words, I am blind only give me important information and use metric units, do not mention that anything about me being blind, use the image if its required: {recognized_text}"
    blocking_future = loop.run_in_executor(
        blocking_executor,
        functools.partial(_run_gemini_generate_sync, text, image_bytes)
    )

    try:
        response = await asyncio.wait_for(blocking_future, timeout=MODEL_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        print("Gemini call timed out")
        await text_to_speech("Request to server timed out")
        return
    except asyncio.CancelledError:
        print("google_searching_with_image cancelled")
        return
    except Exception as e:
        print(f"Gemini error: {e}")
        await text_to_speech("Error from server, please try again later")
        return

    if cancel_event.is_set():
        return

    try:
        answer = response.text
    except Exception as e:
        print(f"Bad Gemini response: {e}")
        return

    if answer:
        await text_to_speech(answer)
                           
## Task 5 -> Help
async def help_function():    
    global vision_model, speech_task_event, speech_task
    
    if vision_model:
        model_running = "meta-llama/llama-4-maverick-17b-128e-instruct"
        vision_model = 0
    else:
        model_running = "meta-llama/llama-4-scout-17b-16e-instruct"
        vision_model = 1
    
    text = f"This photo gives context, I am blind what sort of information do you think I will need and please provide me with that information in less than 50 words do not mention anything about me being blind"
    play_loading_sound("waiting_sound.wav")

    response = await send_to_model(model_running, text)

    if response == None:
        speech_task = asyncio.create_task(text_to_speech("Error from server, please try again later"))
    
    else:
        while speech_task_event.is_set():
            await asyncio.sleep(0.1)
        speech_task = asyncio.create_task(text_to_speech(response))
    

###############################################################
# ------------------------- Flask Trigger---------------------#
###############################################################
async def watch_flask_trigger():
    global state, image_bytes, recognized_text, app_text, speech_task, speech_task_event, recording_task, loop, groq_task, google_task
    
    while True:
        try:
            await received_text_from_app.wait()
            received_text_from_app.clear()
            
            print("Flask Trigger")
            
            if app_text == "start" and state == 0:
                await handle_main_button(loop)
            elif app_text == "stop" and state == 1:
                await handle_main_button(loop)
            elif app_text == "cancel":
                await handle_cancel_button()
            elif app_text != "":
                cancel_event.clear()
                record_stop_event.clear()
                speech_task_event.clear()

                recognized_text = app_text.strip()
                
                task = task_selection()
                print("Task: ", task)
                    
                if task == "1":
                    searching_task = asyncio.create_task(object_searching_async(recognized_text))
                elif task == "2" or task == "4":
                    groq_task = asyncio.create_task(image_description())
                elif task == "3":
                    google_task =  asyncio.create_task(google_searching(recognized_text.strip()))
                elif task == "5":
                    groq_task = asyncio.create_task(help_function())
                state = 0

            app_text = ""
        except Exception as e:
            print(f"Exception: {e}")
            continue
                

# ---------- Button Handlers ---------------
async def handle_main_button(loop):
    global state, image_bytes, recognized_text, recording_task, speech_task, searching_task, speech_task_event, groq_task, google_task

    if state == 0:
        print("Starting voice input...")
        tasks_to_cancel = [recording_task, searching_task, google_task, groq_task]
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to cancel (with short timeout)
        for task in tasks_to_cancel:
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    print(f"Error cancelling task: {e}")

        filename = os.path.dirname(__file__) + "/loading.mp3"
        play_sound_effect(filename)
        state = 1
        cancel_event.clear()
        record_stop_event.clear()
        recording_task = asyncio.create_task(record_audio())

    elif state == 1:
        record_stop_event.set()
        filename = os.path.dirname(__file__) + "/complete.mp3"
        play_sound_effect(filename)
        await asyncio.sleep(1)
        await recording_task

        if cancel_event.is_set():
            speech_task_event.clear()
            print("Recording cancelled")
            state = 0
            return
        
        play_loading_sound("waiting_sound.wav")
        filename = os.path.dirname(__file__) + WAVE_OUTPUT_FILENAME
        
        with open(filename, "rb") as file:
            transcription = client_groq.audio.transcriptions.create(
                  file=file, # Required audio file
                  model="whisper-large-v3-turbo", # Required model to use for transcription
                  prompt="Specify context or spelling",  # Optional
                )
            recognized_text = transcription.text
        
        print(f"Recognised text: {recognized_text}")
        
        task = task_selection().strip()
        print("Task: ", task)
        
        if not recognized_text.strip():
            print("No text found")
            state = 0
            return
            
        if task == "1":
            searching_task = asyncio.create_task(object_searching_async(recognized_text))
        elif task == "2" or task == "4":
            groq_task = asyncio.create_task(image_description())
        elif task == "3":
            google_task =  asyncio.create_task(google_searching(recognized_text.strip()))
        elif task == "5":
            groq_task = asyncio.create_task(help_function())
        state = 0

async def handle_cancel_button():
    global state, recording_task, speech_task, searching_task, google_task, groq_task
    print("Cancel pressed. Aborting operation.")
    cancel_event.set()
    record_stop_event.set()

    # Force stop all audio immediately
    safe_stop_audio()
    
    # Give a moment for audio to fully stop
    await asyncio.sleep(0.2)
    
    tasks_to_cancel = [recording_task, searching_task, google_task, groq_task]
    
    for task in tasks_to_cancel:
        if task and not task.done():
            task.cancel()
    
    # Wait for tasks to cancel (with short timeout)
    for task in tasks_to_cancel:
        if task and not task.done():
            try:
                await asyncio.wait_for(task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                print(f"Error cancelling task: {e}")

    # Cancel speech task with proper cleanup
    if speech_task and not speech_task.done():
        speech_task.cancel()
        try:
            if isinstance(speech_task, asyncio.Task):
                await speech_task
            elif isinstance(speech_task, concurrent.futures.Future):
                await asyncio.wrap_future(speech_task)
        except asyncio.CancelledError:
            print("Speech cancelled")
        except Exception as e:
            print(f"Error cancelling speech: {e}")

        # Now play cancel sound - it should have clear access to audio resources
    filename = os.path.dirname(__file__) + "/cancel_sound.mp3"
    play_sound_effect(filename)
    
    state = 0

# ---------- GPIO Button Watching ----------

def watch_gpio_button(loop):
    # BCM pin numbers (adjust if needed)
    main_button = Button(4, pull_up=True, bounce_time=0.05)    # start/stop
    cancel_button = Button(22, pull_up=True, bounce_time=0.05)  # cancel

    def main_button_pressed():
        print("Main button pressed")
        asyncio.run_coroutine_threadsafe(handle_main_button(loop), loop)

    def cancel_button_pressed():
        print("Cancel button pressed")
        asyncio.run_coroutine_threadsafe(handle_cancel_button(), loop)

    main_button.when_pressed = main_button_pressed
    cancel_button.when_pressed = cancel_button_pressed

    # This blocks to keep the buttons alive, so it should run in its own thread
    pause()

# ---------- Main Entry --------------------
async def main():
    global loop, state, recording_task, cancel_event, record_stop_event, voice
        
    loop = asyncio.get_running_loop()

    # gpio_thread = threading.Thread(target=watch_gpio_button, args=(loop, "/dev/gpiochip0", [2, 3]), daemon=True)
    gpio_thread = threading.Thread(target=watch_gpio_button, args=(loop,), daemon=True)
    gpio_thread.start()
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    saved_mac = os.path.join(script_dir, "saved_mac.txt")
    
    if os.path.getsize(saved_mac) == 0:
        print("No saved device")
    else:
        with open(saved_mac, 'r') as file:
            content = file.read()
            bt_status = pair_trust_connect(content) 

            if bt_status:
                # Schedule the coroutine in the main event loop from this thread
                if loop:
                    await asyncio.sleep(1)
                    asyncio.run_coroutine_threadsafe(text_to_speech("Connected to device"), loop)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
 
    state = 0
    recording_task = None
    cancel_event.clear()
    record_stop_event.clear()

    try:
        await watch_flask_trigger()
    except Exception as e:
        print(f"GPIO error: {e}")

def cleanup_on_exit():
    """Cleanup function to be called on program exit"""
    print("Cleaning up audio resources...")
    safe_stop_audio()

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    print(f"Received signal {signum}, shutting down gracefully...")
    cleanup_on_exit()
    os._exit(0)

if __name__ == "__main__":
    # Register cleanup functions
    atexit.register(cleanup_on_exit)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        cleanup_on_exit()
