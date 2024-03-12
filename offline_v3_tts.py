from transformers import VitsModel, AutoTokenizer
import torch
import scipy

model = VitsModel.from_pretrained("facebook/mms-tts-hun")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hun")

text = "Csillagok, csillagok mondjátok el nekem Merre jár, hol lehet most a kedvesem Veszélyes út, amin jársz, veszélyes út, amin járok Egyszer te is hazatalálsz, egyszer én is hazatalálok, Csillagok, csillagok mondjátok el nekem Merre jár, hol lehet most a kedvesem Veszélyes út, amin jársz, veszélyes út, amin járok Egyszer te is hazatalálsz, egyszer én is hazatalálok"
inputs = tokenizer(text, return_tensors="pt")

print(text)

with torch.no_grad():
    output = model(**inputs).waveform

print(text,output)

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.T.numpy())
