import torch.nn as nn
import torch
from models.audio_model import AudioModel
from models.text_model import TextModel
from models.vision_models import VideoModel, LSTM
from utils.utils import load_config
import torch.nn.init as init

config = load_config('configs/configs.yaml')

class MultimodalClassifier(nn.Module):
    """Class for MultiModal Classification with dynamic model based on 
    modaliities used
    """
    def __init__(self, modalities, hidden_dim=64, output_dim=2):
        """
        Inputs:
            modalities: list of modalities, example:["AUD", "VID", "TEXT"]
            hidden_dim: size of features from each modality(assuming same size output for each mod) 
        """
        super(MultimodalClassifier, self).__init__()

        self.modalities = modalities
        self.models = nn.ModuleDict()

        for modality in modalities:
            if modality == 'VID':
                self.models[modality] = VideoModel(num_hidden_layers=config["VID_HIDDEN_LAYERS"])
                # self.models[modality] = LSTM()
            elif modality == 'TEXT':
                self.models[modality] = TextModel(num_hidden_layers=config["TEXT_HIDDEN_LAYERS"])
            elif modality == 'AUD':
                self.models[modality] = AudioModel(num_hidden_layers=config["AUD_HIDDEN_LAYERS"])

        # Final classification layer
        fusion_layer_input_dim = len(modalities) * hidden_dim
        self.fc = nn.Linear(fusion_layer_input_dim, output_dim)
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.fc.weight)

    def forward(self, X):
        modalities_outputs = []

        for modality in self.modalities:
            modality_input = X[modality + '_features']
            modality_output = self.models[modality](modality_input)
            modalities_outputs.append(modality_output)

        combined_input = torch.cat(modalities_outputs, dim=1)
        output = self.fc(combined_input)
        return output
