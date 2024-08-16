from transformers import AutoProcessor, AutoModelForCTC
processor = AutoProcessor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model = AutoModelForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
print('chinese wav2vec gpt model downloaded')