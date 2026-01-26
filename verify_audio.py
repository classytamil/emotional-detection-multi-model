import torch
import torchaudio
print("Torch version:", torch.__version__)
print("Torchaudio version:", torchaudio.__version__)
try:
    import torchaudio.backend.soundfile_backend
    print("Soundfile backend importable")
except Exception as e:
    print("Backend import error:", e)
print("Success")
