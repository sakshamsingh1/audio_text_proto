# ZS_audio_text
Official release of the INTERSPEECH-23 paper : [A multimodal prototypical approach for unsupervised sound classification](https://arxiv.org/pdf/2306.12300.pdf)

### Environment setup
```
pip install -r requirements.txt
```

### Usage
To use the repo, there are four steps:
1. Clone the repo with submodule
2. Download the data and put them in the `data/input` directory
3. Download pretrained model for AudioClip and LIAON-CLAP
4. Extract the embeddings using the `extract_embed.py` script
OR use the already extracted embeddings
4. For `proto-ac` and `proto-lc` models run `prototypical.py` script with the desired model and dataset

#### Clone the repo
```
git clone --recurse-submodules git@github.com:sakshamsingh1/ZS_audio_text.git
```

#### Download the data
```
cd data/input

# Download the ESC-50 dataset
git clone git@github.com:karolpiczak/ESC-50.git

# Download the US8K dataset
python download_us8k.py

# Download the FSD50K dataset
python download_fsd50k.py
```

#### Download the pretrained models
```
# For AudioCLIP
# should be downloaded in scripts/ref_repo/AudioCLIP/assets
! wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
! rm bpe_simple_vocab_16e6.txt.gz
! wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz 

#FOR LAION_CLAP
# Should be downloaded in data/input
wget https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt 
```

#### Extracting embeddings
This code is slow (has to be optimized) and we provide the extracted embeddings inside `data/processed`
```
python extract_embd.py --model_type <audioclip/clap> --dataset_name <esc50/us8k/fsd50k>
```

#### Our prototypical approach
```
python prototypical.py --model_type <proto-lc/proto-ac> --data <esc50/us8k/fsd50k> --train_type <zs/sv>
```

### Directory structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── data
  ├── input (contains the input data)
       ├── ESC-50 
       ├── US8K
       ├── FSD50K
       ├── Pre-trained models             
  ├── processed (contains the processed data)
       ├── audioClip-feat-embd 
       ├── clap-feat-embd
       ├── other-data
├── scripts (contains the source code)
  ├── ref-repo (contains the reference repo)
```
