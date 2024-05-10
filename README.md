# Multimodal Hate Video Classification
### Project for CSE676:Deep Learning at University at Buffalo, NY (Taught by Dr. Kaiyi Ji)


##### Key Features:   
Modalities and associated large pre-trained model for modal extraction:
1. Video: ViT 
1. Text: BERT
1. Audio: MFCC

Model parametrs can be changed using configs/config.yaml.       

Key configs: 
- MODALILTIES: "VID+AUD+TEXT" (or any combination like "VID")
- TEXT_HIDDEN_LAYERS: 1 (total layers = TEXT_HIDDEN_LAYERS + 2 (first & last fixed))
- AUD_HIDDEN_LAYERS: 1
- VID_HIDDEN_LAYERS: 1
- USE_SCHEDULER: 1
- USE_XAVIER_INIT: 1
- USE_RESIDUAL_BLOCKS: 1 (IF USE_RESIDUAL_BLOCKS is 1, then double the number of hidden layers are effectively applied, since every residual block has 2 linear layers)

Detailed report can be found at report/report.pdf.

##### Sample Notebooks
Sample model training notebook available in notebooks/z3_audio_experiment.ipynb.     
Fusion layer adapts automatically based on the modalities provided in config.yaml file. So test_model_multimodal function can be used for experiments on any modality combination.     

    

###### FAQ

1. Code fails in Windows.    
Solution: The issue is because the code uses paths compatible with Linux/MacOS. Most paths can be updated from configs.yaml file but a few paths inside the code could also be the reason. Debugging is pretty quick if you just run the code in debug mode and follow where the code goes to find all path declarations. 

1. . Can I get the data to run the code?
Solution: You can download the data associated with [HateMM: A Multi-Modal Dataset for Hate Video Classification](https://arxiv.org/abs/2305.03915). Notebooks have the code for feature extraction using pretrained models. However, that can be time-consuming and requires a GPU. If you're just looking to run our test_model_multimodal function, reach out to me at [shailesh.s.mahto@gmail.com](mailto:shailesh.s.mahto@gmail.com). Happy to share extracted features.


