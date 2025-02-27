# https://www.youtube.com/watch?v=L4klnZ5Lox8
import torch
import torchaudio
from ChatTTS import Chat

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

chat = Chat()
chat.load(compile=False)
# torch.manual_seed(1000)

rand_spk = chat.sample_random_speaker()
print(rand_spk)

params_infer_code = Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker
    temperature = 0.5, # using custom temperature
)

inputs_en = """
The true size of india.
""".replace('\n', '')

# use oral_(0-9), laugh_(0-2), break_(0-7)
params_refine_text = Chat.RefineTextParams(
    prompt='[oral_4][laugh_1][break_4]',
)

audio_array_en = chat.infer(
    inputs_en,
    skip_refine_text=True,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code
)

try:
    torchaudio.save("word_level_output.wav", torch.from_numpy(audio_array_en[0]).unsqueeze(0), 24000)
except:
    torchaudio.save("word_level_output.wav", torch.from_numpy(audio_array_en[0]), 24000)