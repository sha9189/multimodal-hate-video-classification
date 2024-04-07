import os
import cv2
from tqdm import tqdm
from multiprocessing import Process, Queue

FOLDER_NAME = 'D:/Buffalo/2024Spring/CSE676/HateMM/Dataset/non_hate_videos/non_hate_videos/'
target_folder = 'D:/Buffalo/2024Spring/CSE676/HateMM/Dataset/non_hate_videos/Dataset_Images/'
folder1 = [""]

def target(queue, video_path, video_folder, frame_count):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        success, img = vidcap.read()
        if not success:
            break
        frame_filename = os.path.join(video_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_filename, img)
        if count % 100 == 0:
            print(f"Processing frame {count} of {video_path}")
        count += 1
        if count > frame_count:
            break
    vidcap.release()
    queue.put(True)

def process_video(video_path, video_folder, frame_count):
    queue = Queue()
    p = Process(target=target, args=(queue, video_path, video_folder, frame_count))
    p.start()
    p.join(60)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"Timeout exceeded for video: {video_path}")

def main():
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for subDir in folder1:
        full_path = os.path.join(FOLDER_NAME, subDir)
        print(f"Processing directory: {full_path}")
        for f in tqdm(os.listdir(full_path)):
            if f.endswith('.mp4'):
                print(f"Processing file: {f}")
                video_path = os.path.join(full_path, f)
                video_folder = os.path.join(target_folder, f.split('.')[0])
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                if os.listdir(video_folder):
                    continue
                vidcap = cv2.VideoCapture(video_path)
                frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                vidcap.release()
                if frame_count == 0:
                    print(f"{f} is an audio file.")
                    audio_folder = os.path.join(target_folder, f.split('.')[0] + " (This is an audio file)")
                    if not os.path.exists(audio_folder):
                        os.makedirs(audio_folder)
                else:
                    process_video(video_path, video_folder, frame_count)

    print("Processing complete.")

if __name__ == '__main__':
    main()
