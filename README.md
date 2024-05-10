# Multimodal Hate Video Classification
### Project for CSE676:Deep Learning at University at Buffalo, NY (Taught by Dr. Kaiyi Ji)

Multimodal Hate Video Classification
     
Modalities and associtaed large pre-trained model for modal extraction:
1. Video: ViT 
1. Text: BERT
1. Audio: MFCC

Model parametrs can be changed using configs/config.yaml. 
Sample model training notebook available in notebooks/z3_audio_experiment.ipynb. 
Fusion layer adapts automatically based on the modalities provided in config.yaml file. So test_model_multimodal function can be used for experiments on any modality combination.

Detailed report can be found at report/report.pdf.
