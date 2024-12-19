import torch
import torchaudio
from ChatTTS import Chat
from datetime import datetime

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

chat = Chat()
chat.load(compile=False)

texts = [
    "Yes, this is a test of the emergency broadcast system."
]

wavs = chat.infer(texts)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"output_{current_time}.wav"

try:
    torchaudio.save(output_filename, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
    print(f"Audio saved successfully as {output_filename}")
except TypeError as e:
    print(f"Encountered error while saving audio: {e}. Retrying without unsqueeze...")
    torchaudio.save(output_filename, torch.from_numpy(wavs[0]), 24000)
    print(f"Audio saved successfully as {output_filename} (retry without unsqueeze).")
except Exception as e:
    print(f"Failed to save audio: {e}")