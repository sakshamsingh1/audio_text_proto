# ZS_audio_text
Official release of multimodal prototypical approach for sound recognition

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

### Installation
```
pip install -r requirements.txt
```

### Usage
To use the repo, there are four steps:
1. Clone the repo with submodules 
2. Download the data and put them in the `data/input` directory
3. Extract the embeddings using the `script/extract_embed.py` script
4. Run the `script/main.py` script with the desired model and dataset

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
import soundata
data_path = 'US8k/'
dataset = soundata.initialize('urbansound8k', data_home=data_path)
dataset.download()

# Download the FSD50K dataset
import soundata
data_path = 'FSD50K/'
dataset = soundata.initialize('fsd50k', data_home=data_path)
dataset.download()
```

#### Download the pretrained models
```
# For AudioCLIP
cd ref-repo/AudioCLIP/assets
! wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
! wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz 
! gunzip ref-repo/AudioCLIP/assets/bpe_simple_vocab_16e6.txt.gz

# For CLAP
! gdown 1Ni8lZ2pryTESjgq8gELLQNM_HGdWtFrE
```

#### TODO
- [x] Download
- [x] add submodules
- [x] add pretrained models
- [ ] add scripts


