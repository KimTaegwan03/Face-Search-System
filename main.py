from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil

from utils.utils import *
import base64
from io import BytesIO
from PIL import Image
from collections import defaultdict

def convert_imgs_to_base64_grouped(image_entries):
    grouped = defaultdict(list)
    for entry in image_entries:
        speaker = entry["speaker"]
        img = entry["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img[..., ::-1])  # BGR to RGB

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        grouped[speaker].append(encoded)
    return grouped


def convert_imgs_to_base64(image_entries):
    result = []
    for entry in image_entries:
        speaker = entry["speaker"]
        img = entry["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img[..., ::-1])  # BGR → RGB

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result.append({
            "speaker": speaker,
            "image": encoded
        })
    return result

app = FastAPI()

ORCH_DIR = "/mnt/share65/orch_jsons"
ASD_DIR = "/mnt/share65/asd_jsons"
VIDEO_DIR = "/mnt/share65/videos"

templates = Jinja2Templates(directory="templates")
app.mount(VIDEO_DIR, StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 미리 지정된 영상 경로
PREDEFINED_VIDEOS = [
    {"id": "--1Hln74Ano", "name": "Kamala Harris is 'cooked' on the national political scene: Charlie Hurt", "path": f"{VIDEO_DIR}/--1Hln74Ano.mp4"},
    {"id": "-3qdllDsGm4", "name": "Trump Gives TRUMPIEST Performance Ever at a FUNERAL & George Santos Addresses Drag Queen Video", "path": f"{VIDEO_DIR}/-3qdllDsGm4.mp4"},
    {"id": "video3", "name": "Video 3", "path": f"{VIDEO_DIR}/video3.mp4"},
]

@app.get("/", response_class=HTMLResponse)
async def select_video_form(request: Request):
    return templates.TemplateResponse("main.html", {
        "request": request,
        "videos": PREDEFINED_VIDEOS
    })

@app.post("/submit/")
async def submit_video(
    request: Request,
    selected_video: str = Form(...),
    custom_video: UploadFile = File(None)
):
    if selected_video == "custom" and custom_video:
        upload_path = os.path.join(UPLOAD_DIR, custom_video.filename)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(custom_video.file, buffer)
        return {"message": "Custom video uploaded", "path": upload_path}
    else:
        matched = next((v for v in PREDEFINED_VIDEOS if v["id"] == selected_video), None)
        if matched:
            video_path = matched["path"]
            basename = os.path.basename(video_path).replace(".mp4","")
            orch_file_path = os.path.join(ORCH_DIR,basename+".json")
            asd_file_path = os.path.join(ASD_DIR,basename+".json")

            overlap = merge_asr_asd(orch_file_path,asd_file_path)

            merged_script = merge_words_by_speaker(orch_file_path,sentence_level=False)

            imgs = extract_faces_from_video(video_path, overlap)

            encoded_imgs = convert_imgs_to_base64_grouped(imgs)

            return templates.TemplateResponse("result.html", {
                "request": request,
                "message": "Predefined video selected",
                "merged_script": merged_script,
                "extracted_imgs": encoded_imgs,
                "custom": False
            })
        else:
            return {"error": "Invalid selection"}
