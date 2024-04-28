import torch
import torch.nn as nn

#adjust unimodels for concate
class audioModel(nn.Module):
    def __init__(self):
        super(audioModel, self).__init__()
        self.fullyconnected_layer1 = nn.Linear(40, 128)
        self.bn1 = nn.BatchNorm1d(128) 
        self.fullyconnected_layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fullyconnected_layer1(x)))
        x = torch.relu(self.bn2(self.fullyconnected_layer2(x)))
        return x

class textModel(nn.Module): 
    def __init__(self):
        super().__init__()
        self.fullyconnected_layer1 = nn.Linear(768, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fullyconnected_layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fullyconnected_layer1(x)))
        x = torch.relu(self.bn2(self.fullyconnected_layer2(x)))
        return x

class videoModel(nn.Module):   # LSTM
    def __init__(self, no_of_frames=100):
        super(videoModel, self).__init__()
        self.lstm = nn.LSTM(768, 128, batch_first=True)
        self.fullyconnected_layer = nn.Linear(128, 64)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fullyconnected_layer(x)
        return x


class combinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = textModel()
        self.video_model = videoModel()
        self.audio_model = audioModel()
        self.fc_fusion_layer = nn.Linear(3 * 64, 192)
        self.classification_head = nn.Linear(192, 2)

    def forward(self, x_text, x_vid, x_audio):
        text_out = self.text_model(x_text)
        video_out = self.video_model(x_vid)
        audio_out = self.audio_model(x_audio)
        concatenated = torch.cat((text_out, video_out, audio_out), dim=1)
        fused = torch.relu(self.fc_fusion_layer(concatenated))
        res = self.classification_head(fused)
        return res


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

# text = textModel()
# text.to(device)
# video = videoModel()
# video.to(device)
# audio = audioModel()
# audio.to(device)
# comb = combinedModel().to(device)