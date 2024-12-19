import torch
import torchaudio
from ChatTTS import Chat
from datetime import datetime

chat = Chat()
chat.load(compile=False)

# rand_spk = chat.sample_random_speaker()
# print(rand_spk)

for i in range(10):
    chat = Chat()
    chat.load(compile=False)
    rand_spk = chat.sample_random_speaker()
    print(i)
    print(rand_spk)