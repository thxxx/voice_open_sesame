apt-get update
apt-get install -y ffmpeg
pip install --upgrade pip
pip install flask flask-cors --ignore-installed
pip install openai fastapi uvicorn[standard] orjson
pip install faster-whisper soundfile librosa
pip install --upgrade transformers accelerate
pip install pydub hf_transfer

git config user.email zxcv05999@naver.com
git config user.name thxxx

pip install torchaudio numpy lhotse huggingface_hub safetensors tensorboard vocos opuslib
pip install cn2an inflect s3tokenizer diffusers conformer pykakasi resemble-perth
pip install pkuseg

# Tokenization
pip install jieba piper_phonemize pypinyin
pip install "setuptools<81"

pip install --upgrade "pyzmq<26"
pip install --upgrade --force-reinstall transformers
pip uninstall -y torchvision
python -m pip install --no-cache-dir "numpy<2.0"

# uvicorn companionserver:app --host 0.0.0.0 --port 5000