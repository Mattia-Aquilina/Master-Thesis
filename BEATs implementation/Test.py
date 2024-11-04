import torch
from tqdm import tqdm
import torchaudio
import os
from BEATs import BEATs, BEATsConfig

THRESHOLD = 0.85

#load model
checkpoint = torch.load('BEATs_iter3_plus_AS2M.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])

#cosine similarity function

def ComputeSimilarity(t1, t2):
    if t1.shape != t2.shape:
        # Here you can handle the mismatched dimensions.
        # Example: If rep1 and rep2 have different lengths, you might need to truncate or pad them.
        # This is just an example. Adjust according to your specific case.
        min_length = min(t1.shape[1], t2.shape[1])
        t1 = t1[:, :min_length]
        t2 = t2[:, :min_length]
    _sim = torch.nn.functional.cosine_similarity(t1, t2)
    return(_sim)

#Constants
REF_SET = "Audio/RealRefrences/"
AUDIO = "./Audio/fake.mp3"


x = torchaudio.load(uri=AUDIO, format="mp3")
rep_x = BEATs_model.extract_features(x[0])[0]

SIM = []

for file in tqdm(os.listdir(REF_SET), desc="Processing refrence set files"):
    audio_name = os.path.join(REF_SET, file)
    sample = torchaudio.load(uri= audio_name,format=os.path.splitext(file)[1])

    audio_rep = BEATs_model.extract_features(sample[0])[0]

    sample = None
    try:
        SIM.append(ComputeSimilarity(rep_x, audio_rep))
    except:
        print("Could not compute similiraty with " + file)
        continue

all_similarities = torch.cat(SIM)

# Find the maximum similarity value
max_similarity = torch.max(all_similarities)

print(max_similarity)