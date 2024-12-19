# import ChatTTS
# import torch
# import torchaudio
#
# chat = ChatTTS.Chat()
# chat.load(compile=False) # Set to True for better performance
#
# texts = ["Use print(torchaudio.list_audio_backends()) to see which backends are available in your current environment. If it returns an empty list, no backend is detected.", "For other systems or if you're not using Conda, download FFmpeg from its official site and add it to your system PATH."]
#
# wavs = chat.infer(texts)
#
# for i in range(len(wavs)):
#     """
#     In some versions of torchaudio, the first line works but in other versions, so does the second line.
#     """
#     try:
#         torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
#     except:
#         torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)




import torch
import torchaudio
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS

chat = ChatTTS.Chat()
chat.load(compile=False)

texts = [
    "Yes, this is a test of the emergency broadcast system.",
    ]

wavs = chat.infer(texts)

try:
    torchaudio.save(f"basic_output{0}.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
except:
    torchaudio.save(f"basic_output{0}.wav", torch.from_numpy(wavs[0]), 24000)