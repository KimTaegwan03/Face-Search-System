import os
import json
import av
import cv2

'''
화자 분리 json 파일과 ASD json 파일을 병합하는 파일

merge_asr_asd : asd 파일의 object 기준으로 iterate하며 화자 분리 object와 겹치는 시간 구간의 화자를 할당해주는 함수
extract_faces_from_video : 화자별로 이미지를 크롭하고 저장하는 함수
'''

def merge_asr_asd(orch_path, asd_path):
    if not os.path.exists(orch_path) or not os.path.exists(asd_path):
        return
    
    with open(orch_path,"r") as f:
        orch_file = json.load(f)

    with open(asd_path,"r") as f:
        asd_file = json.load(f)

    curr_orch_idx = 0

    overlap_list = []

    for curr_asd in asd_file:
        asd_start = curr_asd['start']
        asd_end = curr_asd['end']

        # orch_file 끝까지 검사
        while curr_orch_idx < len(orch_file):
            orch = orch_file[curr_orch_idx]
            orch_start = orch['start']
            orch_end = orch['end']

            # orch가 현재 asd보다 완전히 앞에 있음 → orch 인덱스 증가
            if orch_end <= asd_start:
                curr_orch_idx += 1
                continue

            # orch가 현재 asd보다 완전히 뒤에 있음 → 이 asd는 처리 안 함
            if orch_start >= asd_end:
                break

            # 겹치는 경우 처리
            print(f"Overlap Detected:\n - ASD:  {asd_start:.2f} ~ {asd_end:.2f}\n - ORCH: {orch_start:.2f} ~ {orch_end:.2f}")
            
            # 필요한 작업 수행
            overlap_list.append({'speaker':orch_file[curr_orch_idx]['speaker'],'time':asd_start,'bboxes':curr_asd['bboxes']}) # speaker, time(sec), bbox

            break  # 겹치면 해당 ASD만 처리하고 다음 ASD로 넘어감

    return overlap_list

def extract_faces_from_video(video_path, overlap_list):

    entries = overlap_list

    # 영상 불러오기
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)

    output = []

    for entry in entries:
        speaker = entry['speaker']
        time_sec = entry['time']
        bbox = entry['bboxes']
        x1, y1, x2, y2 = map(int, bbox)

        # 해당 시간의 프레임 번호 계산
        frame_idx = int(time_sec * fps)

        # 정확한 프레임 seek
        container.seek(int(time_sec * av.time_base))  # time_base = 1e6
        frame = None
        for packet in container.demux(stream):
            for frm in packet.decode():
                if frm.pts * frm.time_base >= time_sec:
                    frame = frm.to_ndarray(format='bgr24')
                    break
            if frame is not None:
                break
        
        if frame is None:
            print(f"[WARN] Could not extract frame at {time_sec}s")
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            print(f"[WARN] Empty crop for {speaker} at {time_sec}s")
            continue

        output.append(
            {"speaker":speaker,
             "image":face_crop}
        )

    container.close()

    return output

def merge_words_by_speaker(orch_path, sentence_level=False):
    import regex as re
    import json
    """
    같은 화자의 연속된 단어들을 문장 단위로 병합하고, speaker를 숫자 ID로 변환.

    Parameters:
        word_list (list): 각 단어에 대한 정보가 담긴 딕셔너리 리스트

    Returns:
        list: 문장 단위로 병합된 딕셔너리 리스트 (speaker는 int ID)
    """
    with open(orch_path,"r") as f:
        word_list = json.load(f)

    def speaker_to_id(speaker_str):
        match = re.search(r'\d+', speaker_str)
        return int(match.group()) if match else -1

    sentence_endings = {".", "?", "!"}
    merged = []
    current = {
        "speaker": speaker_to_id(word_list[0]["speaker"]),
        "word": word_list[0]["word"],
        "start": word_list[0]["start"],
        "end": word_list[0]["end"]
    }

    for word_info in word_list[1:]:
        speaker_id = speaker_to_id(word_info["speaker"])
        word = word_info["word"]
        is_sentence_end = word and word[-1] in sentence_endings

        if speaker_id == current["speaker"]:
            current["word"] += " " + word_info["word"]
            current["end"] = word_info["end"]

            if sentence_level and is_sentence_end:
                merged.append(current)
                current = {
                    "speaker": speaker_id,
                    "word": "",
                    "start": word_info["end"],  # 다음 문장은 이후 시간부터 시작
                    "end": word_info["end"]
                }

        else:
            merged.append(current)
            current = {
                "speaker": speaker_id,
                "word": word_info["word"],
                "start": word_info["start"],
                "end": word_info["end"]
            }
    if current["word"]:
        merged.append(current)

    return merged


if __name__ == "__main__":
    json_dir = "/mnt/share65/orch_jsons"
    video_dir = "/mnt/share65/videos"
    # audio_dir = "data/audio"
    feature_dir = "/mnt/share65/emb"
    graph_dir = "/mnt/share65/emb/graph"

    orch_path = os.path.join(json_dir,"Ez6phBoRvpc.json")
    video_path = os.path.join(video_dir,"Ez6phBoRvpc.mp4")
    asd_path = "Ez6phBoRvpc.json"

    ret = merge_asr_asd(orch_path,asd_path,video_path)

    # print(json.dumps(ret,indent=4))

    imgs = extract_faces_from_video(video_path, ret,"spekaer_img")