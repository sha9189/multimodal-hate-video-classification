import os
import matplotlib
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pickle

print(librosa.__version__)
print(matplotlib.__version__)


base_dir = 'D:/Buffalo/2024Spring/CSE676/HateMM/Dataset/audio'
hate_audio_dir = os.path.join(base_dir, 'hate_audio_wav')
non_hate_audio_dir = os.path.join(base_dir, 'non_hate_audio_wav')
hate_waveform_dir = os.path.join(base_dir, 'hate_waveforms')
non_hate_waveform_dir = os.path.join(base_dir, 'non_hate_waveforms')

os.makedirs(hate_waveform_dir, exist_ok=True)
os.makedirs(non_hate_waveform_dir, exist_ok=True)

#saving waveforms as images(pngs)
def save_waveform(waveform, sr, filepath):
    """Function to plot adn save waveform images using wav file"""
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(waveform, sr=sr, color='b')
    plt.title('Waveform')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

#extracting averaged MFCCs and waveform
def extract_features_and_waveform(file_path):
    """Function to take a wav file and convert to mfcc features"""
    try:
        audio, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean, audio, sr
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(40), None, None

data = []

#looping through files
for directory, label in [(hate_audio_dir, 'hate'), (non_hate_audio_dir, 'non_hate')]:
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            mfccs_mean, waveform, sr = extract_features_and_waveform(file_path)
            video_name = os.path.splitext(filename)[0]

            if waveform is not None:
                waveform_image_path = os.path.join(
                    hate_waveform_dir if label == 'hate' else non_hate_waveform_dir,
                    video_name + '.png'
                )
                save_waveform(waveform, sr, waveform_image_path)
            

            data_entry = {
                'mfccs': mfccs_mean,
                'label': label,
                'video_name': video_name
            }
            data.append(data_entry)

#saving the extracted features and metadata to a pickle file
pickle_path = os.path.join(base_dir, 'audio_features_waveforms_labels.pkl')
with open(pickle_path, 'wb') as outfile:
    pickle.dump(data, outfile)
