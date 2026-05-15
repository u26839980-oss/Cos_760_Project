import whisper, gensim, bertopic, sklearn, torch

print('Whisper:', whisper.__version__ if hasattr(whisper, '__version__') else 'ok')
print('Gensim:', gensim.__version__)
print('Bertopic:', bertopic.__version__)
print('Torch:', torch.__version__)
print('GPU available: ', torch.cuda.is_available)