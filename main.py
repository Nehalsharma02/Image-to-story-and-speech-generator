from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uuid
#from unsloth import FastVisionModel
import torch
from PIL import Image
from gtts import gTTS
import os
import io

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory storage (replace with a database in production)
stories = {}
languages = ["en", "es", "fr", "de", "hi", "mr", "kn", "bn", "gu", "pa", "ta", "te", "ml", "ne", "or"]

# Pydantic models
class TranslateRequest(BaseModel):
    storyId: str
    language: str

class TextToSpeechRequest(BaseModel):
    text: str
    language: str

# Model Loading and Setup
# def setup_model(use_cpu=False):
#     device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
#     print(f"Using device: {device}")

#     try:
#         model, tokenizer = FastVisionModel.from_pretrained(
#             "unsloth/Qwen2-VL-2B-Instruct",
#             load_in_4bit=True,
#             use_gradient_checkpointing="unsloth",
#         )
#         FastVisionModel.for_inference(model)
#         return model, tokenizer, device
#     except RuntimeError as e:
#         if "out of memory" in str(e) and device == "cuda":
#             print("GPU out of memory. Switching to CPU.")
#             return setup_model(use_cpu=True)
#         else:
#             raise e

#model, tokenizer, device = setup_model()  # Load model globally

# Image Analysis and Story Generation
def generate_story(image, instruction, device):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)
    output_tokens = model.generate(**inputs, max_new_tokens=2000, use_cache=True, temperature=1.5, min_p=0.1)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

# Text to Speech Conversion
def text_to_speech(text, language='en', filename='output.mp3'):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        return None

# Endpoints

@app.post("/api/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        img_content = await image.read()
        img = Image.open(io.BytesIO(img_content))
        instruction = "Analyze the given image and create a story about that image"
        #generated_text = generate_story(img, instruction, device)
        generated_text = "Sunlight streamed into the cozy room in Indore, illuminating a circle of intent young faces. Spread on the soft rug were colorful picture cards and open books, evidence of a lively learning session. A little girl with a bow in her hair held up a card with a drawing of a friendly-looking animal, her eyes sparkling with discovery. Her companions, two boys and another girl, leaned in with curiosity, their small fingers pointing at the illustrations. Laughter occasionally bubbled up as they shared their observations and guesses. In this warm, child-friendly space in Indore, learning wasn't a chore, but a joyful adventure shared amongst friends."
        story_id = str(uuid.uuid4())
        stories[story_id] = {"originalText": generated_text, "translations": {}}
        return JSONResponse({"storyId": story_id, "story": generated_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/story/{story_id}")
async def get_story(story_id: str):
    if story_id not in stories:
        raise HTTPException(status_code=404, detail="Story not found")
    return stories[story_id]

@app.get("/api/languages")
async def get_languages():
    return languages

@app.post("/api/translate")
async def translate_story(translate_request: TranslateRequest):
    if translate_request.storyId not in stories:
        raise HTTPException(status_code=404, detail="Story not found")
    if translate_request.language not in languages:
        raise HTTPException(status_code=400, detail="Language not supported")

    original_text = stories[translate_request.storyId]["originalText"]
    # In a real app, you would use a translation service here
    translated_text = f"Translated {original_text} to {translate_request.language}" #placeholder
    stories[translate_request.storyId]["translations"][translate_request.language] = translated_text
    return JSONResponse({"translatedText": translated_text})

# @app.post("/api/text-to-speech")
# async def text_to_speech_endpoint(tts_request: TextToSpeechRequest):
#     speech_file = text_to_speech(tts_request.text, tts_request.language)
#     if speech_file:
#         return FileResponse(speech_file, media_type="audio/mpeg", filename="output.mp3")
#     else:
#         raise HTTPException(status_code=500, detail="Failed to generate speech")

@app.post("/api/text-to-speech")
async def text_to_speech_endpoint(tts_request: TextToSpeechRequest):
    filename = "output.mp3"
    filepath = text_to_speech(tts_request.text, tts_request.language, filename=filename)

    if filepath and os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg", filename=filename)
    else:
        return {"error": "Failed to generate audio."}

@app.get("/api/stories")
async def get_saved_stories():
    return stories

@app.get("/health")
async def health_check():
    return {"status": "ok"}


