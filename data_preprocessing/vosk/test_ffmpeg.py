#!/usr/bin/env python3

import subprocess
import sys
import os
import pickle
from tqdm import tqdm

from vosk import Model, KaldiRecognizer, SetLogLevel

SAMPLE_RATE = 16000

SetLogLevel(0)

model_name = "vosk-model-small-en-us-0.15"
# model_name = "vosk-model-en-us-0.22"

model = Model(f"./Codes/vosk/{model_name}")
rec = KaldiRecognizer(model, SAMPLE_RATE)

#ffmpeg_path = "/Users/zhaowendan/Documents/CSE 676 Deep Learning/project/HateMM/venv/lib/python3.12/site-packages/ffmpeg"  
#os.environ['PATH'] += f':{os.path.dirname(ffmpeg_path)}'


def get_transcript(file_path:str):
    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                                file_path,
                                "-ar", str(SAMPLE_RATE) , "-ac", "1", "-f", "s16le", "-"],
                                stdout=subprocess.PIPE) as process:

        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                pass
                # print(rec.Result())
            else:
                # print(rec.PartialResult())
                pass

        transcript = rec.FinalResult()[12:-1]
    return transcript


start_video = 1
last_video = 652
# video_category = "hate_video"
video_category = "non_hate_video"
video_folder = f"Dataset/{video_category}s/"

all_transcripts = ""

for i in tqdm(range(start_video, last_video+1)):
    file_path = f"{video_folder}/{video_category}_{i}.mp4"
    transcript =  get_transcript(file_path)
    with open(f"Dataset/{video_category}_transcripts/{video_category}_{i}.txt", "w") as file:
        file.write(transcript)
