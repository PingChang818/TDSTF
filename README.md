# TDSTF
This is the github repository for the paper "A Transformer-based Diffusion Probabilistic Model for Heart Rate and Blood Pressure Forecasting in Intensive Care Unit" (https://doi.org/10.1016/j.cmpb.2024.108060)

# MIMIC-III data
Download the dataset at

https://physionet.org/content/mimiciii/1.4/

# Environment
[Anaconda3-2023.03-0-Windows-x86_64](https://repo.anaconda.com/archive/)

[Pytorch=2.1.1 + cuda=11.8](https://pytorch.org/)

# Data preprocessing
Create empty folders: "/save", "/preprocess/data", "/preprocess/data/MIMICIII", and "/preprocess/data/first"

Download the MIMIC-III data to "/preprocess/data/MIMICIII"

Run the files step_1.py through step_4.py in order in the folder "/preprocess"

# Experiments
Run the file main.py

To test a pretrained model, please assign the model folder name to the parameter "modelfolder"

# Acknowledgements
A part of the codes is based on [CSDI](https://github.com/ermongroup/CSDI) and [STraTS](https://github.com/sindhura97/STraTS)
