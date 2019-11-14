# GATA: Multi-Theme Generative Adversarial Terrain Amplification

## Synopsis

The repository contains source code supporting the paper "Multi-Theme Generative Adversarial Terrain Amplification" by Yiwei Zhao, Han Liu, Igor Borovikov, Ahmad Beirami, Maziar Sanjabi, and Kazi Zaman (EA Digital Platform – Data & AI, Electronic Arts), accepted at SIGGRAPH Asia 2019.

## Getting started

Start with creating conda environment and installing the necessary packages:

    conda create --name GATA python=3.6
    conda activate GATA
    conda install -c anaconda numpy
    conda install -c menpo opencv
    conda install -c anaconda scipy
    pip install psychopy
    conda install -c anaconda 'tensorflow-gpu=1.10.*'(GPU)
    or
    conda install -c aaronzs tensorflow=1.10(CPU)

Next, clone the repository.

## Dataset

Download the raw dataset to folder `dataset/raw_dataset/`. We will provide some links to the potential sources of publicly available datasets for experimentation.
  
## Preprocessing

Run the following script to build training set:

    python preprocess/clip_all.py
    python preprocess/build_dataset_all.py
 
Note, the package 'psychopy' can only run on windows and Mac machine now.

## Training

Run the following script:

    python training.py --output_dir training_result/
    
## Testing

We are not sharing checkpoints at this moment, you need to run preprocessing and then training from scratch before you can test.

Place the checkpoint files and 'options.json' to the folder `inference/checkpoint/`.

Then run the following script:

    python inference/generate.py
    
Check the comments in `inference/generate.py` for the avaiable inference options.

## Code maintenance

The presented code base is experimental. The authors are aware that the code can benefit from a better structure,
explicit hyperparameters extraction and their explanation. We intend to make such changes in the future. Stay tuned.

## Code base contributors

- Yiwei Zhao
- Han Liu
- Maziar Sanjabi
