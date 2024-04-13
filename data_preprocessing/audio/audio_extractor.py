import os
import argparse
from moviepy.editor import VideoFileClip

def video_audio_extractor(video_path, output_path):
    """Function to extract mp3 files from video files"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_file_path = os.path.join(output_path, 'errors.log')
    for filename in os.listdir(video_path):
        full_video_path = os.path.join(video_path, filename)
        if not filename.lower().endswith(('.mp4')):
            print(f"Non-video file: {filename}")
            continue
        audioname = os.path.splitext(filename)[0]
        audio_output = os.path.join(output_path, f'{audioname}.mp3')

        try:
            video = VideoFileClip(full_video_path)
            video.audio.write_audiofile(audio_output)
            video.close()
        except KeyError as e:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Skipping {filename} due to KeyError: {e}\n")
            print(f"Skipping {filename} due to error: {e}")
        except Exception as e:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Error processing {filename}: {e}\n")
            print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extracting audio')
    parser.add_argument('video_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    video_audio_extractor(args.video_path, args.output_path)

if __name__ == '__main__':
    main()
