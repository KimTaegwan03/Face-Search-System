import os
import sys
import json
import glob
import math
import torch
import av
import numpy as np
import python_speech_features
import cv2
from scipy.io import wavfile
from argparse import ArgumentParser
from model.faceDetector.s3fd import S3FD
from ASD import ASD
import warnings

import sqlite3
import pandas as pd
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

warnings.filterwarnings("ignore")

def convert_video_to_25fps(input_path, output_path):
    command = f"ffmpeg -y -i {input_path} -r 25 -preset ultrafast -an {output_path} -loglevel panic"
    os.system(command)

def extract_audio(video_path, audio_path):
    command = f"ffmpeg -y -i {video_path} -qscale:a 0 -ac 1 -vn -ar 16000 -acodec pcm_s16le {audio_path} -loglevel panic"
    os.system(command)

def extract_mfcc(audio_path):
    sr, audio = wavfile.read(audio_path)
    mfcc = python_speech_features.mfcc(audio, sr, numcep=13)
    return mfcc

def extract_video_frames_pyav(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    container = av.open(video_path)
    frame_buffer = []
    for idx, frame in enumerate(container.decode(video=0)):
        if idx % 25 != 0:
            # frame_buffer.append((idx,[]))
            continue
        img = frame.to_ndarray(format='bgr24')
        frame_buffer.append((idx, img))
    return frame_buffer

def detect_faces(frame_buffer, facedetScale):
    detector = S3FD(device='cuda')
    all_faces = []
    for fidx, img in frame_buffer:
        if fidx % 25 != 0:
            continue
        image_rgb = img[:, :, ::-1]
        bboxes = detector.detect_faces(image_rgb, conf_th=0.9, scales=[facedetScale])
        faces = []
        for bbox in bboxes:
            x1, y1, x2, y2, conf = *bbox[:-1], bbox[-1]
            faces.append({'frame': fidx, 'bbox': [x1, y1, x2, y2], 'conf': conf})
        all_faces.append(faces)
    return all_faces

def extract_face_crops(frame_buffer, faces, crop_scale=0.4):
    frame_dict = {fidx: img for fidx, img in frame_buffer}
    tracks = []
    for frame_faces in faces:
        for face in frame_faces:
            frame_idx = face['frame']
            image = frame_dict[frame_idx]
            x1, y1, x2, y2 = face['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            size = max((x2 - x1), (y2 - y1)) / 2
            size *= (1 + crop_scale * 2)
            mx = int(cx - size)
            my = int(cy - size)
            mx2 = int(cx + size)
            my2 = int(cy + size)
            crop = image[my:my2, mx:mx2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            crop_resized = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (224, 224))
            center_crop = crop_resized[56:168, 56:168]  # 112x112
            tracks.append({
                'start': frame_idx / 25.0,
                'end': (frame_idx + 1) / 25.0,
                'frame': frame_idx,
                'face': center_crop,
                'bbox': [x1, y1, x2, y2]
            })
    return tracks


def run_asd_model(tracks, mfcc, model: ASD):
    tracks.sort(key=lambda x: x['frame'])
    video_input = [t['face'] for t in tracks]
    audio_input = mfcc

    audio_len = (audio_input.shape[0] - audio_input.shape[0] % 4) // 100
    video_len = len(video_input) // 25
    total_length = min(audio_len, video_len)

    duration = 4  # seconds
    stride = 1    # seconds
    sample_rate = 25

    scores = []

    for i in range(0, total_length - duration + 1, stride):
        v_start = i * sample_rate
        v_end = (i + duration) * sample_rate
        a_start = i * 100
        a_end = (i + duration) * 100

        v_chunk = video_input[v_start:v_end]
        a_chunk = audio_input[a_start:a_end]

        inputV = torch.FloatTensor(np.stack(v_chunk)).unsqueeze(0).cuda()
        inputA = torch.FloatTensor(a_chunk).unsqueeze(0).cuda()

        with torch.no_grad():
            v_emb = model.model.forward_visual_frontend(inputV)
            a_emb = model.model.forward_audio_frontend(inputA)
            out = model.model.forward_audio_visual_backend(a_emb, v_emb)
            score = model.lossAV.forward(out, labels=None)
            scores.extend(-score)

    # 실제 score 길이에 맞춰 tracks 자르기
    score_count = len(scores)
    return scores, tracks[:score_count * stride * sample_rate]


def save_json(tracks, scores, out_path):
    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)

        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    results = []
    group = []
    iou_threshold = 0.5

    for i in range(len(tracks)):
        t = tracks[i]
        s = float(scores[i])

        if not group:
            group.append((t, s))
            continue

        prev_bbox = group[-1][0]['bbox']
        curr_bbox = t['bbox']
        iou = compute_iou(prev_bbox, curr_bbox)

        if iou >= iou_threshold:
            group.append((t, s))
        else:
            start_time = group[0][0]['start']
            end_time = group[-1][0]['end']
            duration = end_time - start_time
            if duration >= 1.0:
                mid_idx = len(group) // 2
                start_track, start_score = group[0]
                mid_track, mid_score = group[mid_idx]
                results.append({
                    'start': round(start_time, 3),
                    'end': round(end_time, 3),
                    'bboxes': [round(x, 2) for x in start_track['bbox']],
                    'score': float(np.mean([s for _, s in group]))
                })
            group = [(t, s)]

    # 마지막 그룹도 처리
    if group:
        start_time = group[0][0]['start']
        end_time = group[-1][0]['end']
        duration = end_time - start_time
        if duration >= 1.0:
            mid_idx = len(group) // 2
            start_track, start_score = group[0]
            mid_track, mid_score = group[mid_idx]
            results.append({
                'start': round(start_time, 3),
                'end': round(end_time, 3),
                'bboxes': [round(x, 2) for x in start_track['bbox']],
                'score': float(np.mean([s for _, s in group]))
            })

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)



def visualize_results(frame_dir, tracks, scores, save_path):
    import glob
    color_dict = {True: (0, 255, 0), False: (0, 0, 255)}
    scores = [float(s) for s in scores]
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))

    # 미리 첫 프레임으로 해상도 확인
    test_frame = cv2.imread(frame_paths[0])
    height, width = test_frame.shape[:2]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))

    # frame 단위로 바로 저장
    for i, fpath in enumerate(frame_paths):
        frame = cv2.imread(fpath)
        frame_copy = frame.copy()
        for t, s in zip(tracks, scores):
            if t['frame'] == i:
                x1, y1, x2, y2 = map(int, t['bbox'])
                label = float(s) >= 0
                color = color_dict[label]
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_copy, f"{s:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out.write(frame_copy)

    out.release()


def main():
    import sqlite3
    import pandas as pd
    from tqdm import tqdm

    db_path = "/mnt/share65/speech_segments.db"
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("SELECT video_id FROM video_metadata ORDER BY video_id", conn)

    video_list = df['video_id'].to_list()
    video_list.sort()

    parser = ArgumentParser()
    parser.add_argument('--videoDir', type=str, default='/mnt/share65/videos', help='Path to input video')
    parser.add_argument('--videoDir25fps', type=str, default='/mnt/share65/videos_25fps', help='Path to input video')
    parser.add_argument('--pretrainModel', type=str, default='weight/pretrain_AVA_CVPR.model', help='Path to pretrained ASD model')
    parser.add_argument('--outputDir', type=str, default='/mnt/share65/asd_jsons', help='Path to save output JSON')
    parser.add_argument('--visualizePath', type=str, default='output.avi', help='Path to save annotated video')
    parser.add_argument('--frameDir', type=str, default='frames', help='Directory to store extracted frames')
    parser.add_argument('--facedetScale', type=float, default=0.25)
    args = parser.parse_args()

    # print("[INFO] Converting video to 25 FPS...")
    # converted_video_path = 'converted_25fps.avi'
    # # convert_video_to_25fps(args.videoPath, converted_video_path)
    model = ASD()
    model.loadParameters(args.pretrainModel)
    model.eval()

    os.makedirs(args.outputDir,exist_ok=True)

    for video_id in tqdm(video_list):
        videoPath = os.path.join(args.videoDir,video_id+".mp4")
        videoPath25fps = os.path.join(args.videoDir25fps,video_id+".mp4")
        outputPath = os.path.join(args.outputDir,video_id+".json")

        if os.path.exists(outputPath):
            continue
        audio_path = 'temp_audio.wav'
        extract_audio(videoPath, audio_path)

        mfcc = extract_mfcc(audio_path)

        frame_buffer = extract_video_frames_pyav(videoPath25fps, args.frameDir)

        faces = detect_faces(frame_buffer, args.facedetScale)

        tracks = extract_face_crops(frame_buffer, faces)

        scores, final_tracks = run_asd_model(tracks, mfcc, model)

        save_json(final_tracks, scores, outputPath)

        os.remove(audio_path)

if __name__ == '__main__':
    main()

 
