from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os

Text_FOLDER = '../transcripts/'

transCript = {}

# Loop through each file in the folder
for filename in os.listdir(Text_FOLDER):
    if filename.endswith('.txt'): 
        file_path = os.path.join(Text_FOLDER, filename)
        with open(file_path, 'r') as file:
            # Read text file contents and append to dictionary
            transCript[filename[:-4]] = file.read()

print(len(transCript))

'''
num = len(transCript)
print(num)
sum = 0  
for script in transCript:
   words = script.split()
   sum += len(words)
print(sum // num)
'''

Pf_Folder = '../saved_pickles/'
# Save text contents into a pickle file
with open(Pf_Folder + 'all_video_vosk_audioMap.p', 'wb') as pf:
    pickle.dump(transCript, pf)

with open(Pf_Folder + 'all_video_vosk_audioMap.p','rb') as fp:
    transCript = pickle.load(fp)


from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")



from tqdm import tqdm
allEmbedding ={}
for i in tqdm(transCript):
  try:
    inputs = tokenizer(transCript[i], return_tensors="pt", truncation = True, padding='max_length', add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        allEmbedding[i]= last_hidden_states[0][0].detach().numpy()
    del(outputs)
  except:
    pass



print(len(allEmbedding))
with open(Pf_Folder+'all_rawBERTembedding.p', 'wb') as fp:
    pickle.dump(allEmbedding,fp)

