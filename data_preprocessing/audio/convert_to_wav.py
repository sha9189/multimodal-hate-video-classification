import os
import librosa
import soundfile as sf

def convert_folder_mp3_to_wav(src_folder, dest_folder):
    """Function to convert mp3 file to wav file"""
    os.makedirs(dest_folder, exist_ok=True)
    for file_name in os.listdir(src_folder):
        if file_name.endswith('.mp3'):
            src_path = os.path.join(src_folder, file_name)
            dest_path = os.path.join(dest_folder, os.path.splitext(file_name)[0] + '.wav')
            audio, sr = librosa.load(src_path, sr=None)
            sf.write(dest_path, audio, sr)

src_folder = 'D:/Buffalo/2024Spring/CSE676/HateMM\Dataset/audio/non_hate_audio'
dest_folder = 'D:/Buffalo/2024Spring/CSE676/HateMM/Dataset/audio/non_hate_audio_wav'
convert_folder_mp3_to_wav(src_folder, dest_folder)
