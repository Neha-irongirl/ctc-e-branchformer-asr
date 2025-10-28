# ctc-e-branchformer-asr

# Problem statement:This project implements a CTC-only Automatic Speech Recognition (ASR) system using the
# E-Branchformer encoder, trained on the LibriSpeech 100-hour dataset. The model integrates a
# Language Model (LM) during inference to enhance decoding accuracy and reduce Word Error Rate
# (WER).

ctc-e-branchformer-asr/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ (place LibriSpeech files or links here)
├─ scripts/
│  ├─ build_lm.sh
│  └─ prepare_manifest.py
├─ src/
│  ├─ features.py
│  ├─ model.py
│  ├─ train.py
│  ├─ decode.py
│  └─ evaluate.py
├─ configs/
│  └─ config.yaml
└─ assets/
   └─ (place training curves / sample outputs)


# requirements.txt
torch>=1.12
torchaudio
numpy
scipy
librosa
pyctcdecode
kenlm
jiwer
tqdm
PyYAML
soundfile


# dataset:
  sample_rate: 16000
  n_mels: 80
  n_fft: 512
  win_length: 400
  hop_length: 160

# train:
  batch_size: 16
  epochs: 60
  lr: 1e-4
  device: cuda

# model:
  input_dim: 80
  d_model: 256
  num_layers: 12
  nhead: 4
  conv_expansion: 2
  dropout: 0.1

# decode:
  beam_width: 100
  lm_weight: 2.0
  word_score: -1.0

