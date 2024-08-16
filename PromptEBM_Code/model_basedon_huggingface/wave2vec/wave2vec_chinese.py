import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
'''
https://github.com/fxsjy/jieba
path=/data/dingcheng/workspace/baidu/ccl/Lifelonglearning/Lifelonglearning_For_NLG_v3/model_basedon_huggingface/EBM_NR_LAMOL/common_voice/cv-corpus-8.0-2022-01-19/zh-CN/clips
'''
#test_dataset = load_dataset("common_voice", "zh-CN", split="test")
test_dataset = load_dataset("/data/dingcheng/workspace/baidu/ccl/Lifelonglearning/Lifelonglearning_For_NLG_v3/model_basedon_huggingface/EBM_NR_LAMOL/common_voice/cv-corpus-8.0-2022-01-19/", "zh-CN", split="test")
processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset[:2]["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset[:2]["sentence"])