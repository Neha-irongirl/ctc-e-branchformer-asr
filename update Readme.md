# Problem statement

This project implements a CTC-only Automatic Speech Recognition (ASR) system using the E-Branchformer encoder, trained on the LibriSpeech 100-hour dataset. The model integrates a Language Model (LM) during inference to enhance decoding accuracy and reduce Word Error Rate (WER).
# Repository structure
ctc-e-branchformer-asr/
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
