from espnet_onnx.export import ASRModelExport
from espnet2.bin.asr_inference import Speech2Text as EspnetSpeech2Text
from espnet_onnx import Speech2Text
import librosa
import time
from datasets import load_dataset
from tqdm import tqdm

tag = 'pyf98/librispeech_conformer'
wav_file = '029f6450-447a-11e9-a9a5-5dbec3b8816a.wav'
audio, sr = librosa.load(wav_file)

# espnet load from pretrained
'''espnet_model = EspnetSpeech2Text.from_pretrained(tag)

print('running regular inference')
start_time = time.time()
nbest = espnet_model(audio)
transcript = nbest[0][0]
latency = time.time() - start_time
print(f'transcript={transcript}, latency={latency}')

m = ASRModelExport()
m.export(espnet_model, tag)'''

# onnx export
dataset = load_dataset(
        "patrickvonplaten/librispeech_asr_dummy", "clean", split="validation"
    )

onnx_model = Speech2Text(tag_name=tag)

for example in tqdm(dataset):
    audio = example["audio"]["array"]
    print("audio length",len(audio))
    # onnx inference
    print('running ONNX inference')
    start_time = time.time()
    nbest = onnx_model(audio)
    transcript = nbest[0][0]
    latency = time.time() - start_time
    print(f'transcript={transcript}, latency={latency}')
