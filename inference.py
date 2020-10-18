import torch

from model import VSE
from vocab import Vocabulary
from data import get_test_loader
from evaluation import encode_data

import numpy as np
import nltk
import pickle
import os

device = "cuda"
dataset = "fashion"

## load fine-tuned model
checkpoint = torch.load("./runs/runX/model_best.pth.tar", map_location=torch.device(device))
opt = checkpoint['opt']

# load vocabulary used by the model
with open(f"./vocab/{dataset}_vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)

assert opt.vocab_size == len(vocab)

model = VSE(opt)
model.load_state_dict(checkpoint['model'])

## create embeddings for all images (once)
print('Loading dataset')
print(opt.data_name)
data_loader = get_test_loader("test", opt.data_name, vocab, opt.crop_size,
                            opt.batch_size, opt.workers, opt)

print('Computing results...')
img_embs, cap_embs = encode_data(model, data_loader, on_gpu=True)

if dataset == "flickr":
    ## only keep first 6k unique images (for Flickr)
    mask = np.array([i for i in range(img_embs.shape[0]) if i % 5 == 0])
    img_embs = img_embs[mask]

elif dataset == "fashion":
    ## for fashion keep every 3rd
    mask = np.array([i for i in range(img_embs.shape[0]) if i % 3 == 0])
    img_embs = img_embs[mask]


assert img_embs[0].shape[0] == cap_embs[0].shape[0]

## create embedding for new text query 
query = "A woman climbs a mountain"
query = "blue women skirt printed"

# from PrecomputeDataset: Convert caption (string) to word ids.
tokens = nltk.tokenize.word_tokenize(
    str(query).lower())
caption = []
caption.append(vocab('<start>'))
caption.extend([vocab(token) for token in tokens])
caption.append(vocab('<end>'))
target = torch.tensor(caption, dtype=torch.long).cuda()

txt_emb = model.txt_enc(target.unsqueeze(0), torch.tensor([len(target),]))


## search k most similar img embeddings
txt_emb = txt_emb[0].detach().cpu().numpy()

scores = np.dot(img_embs, txt_emb)
scores.shape

sort_idx = np.argsort(scores)[::-1]
scores[sort_idx[:5]]

# len(data_loader.dataset.captions)

if dataset == "flickr":
    ## multiply index by 5 as there are 5 captions for each image
    for idx in sort_idx[:10]:
        data_loader.dataset.captions[5 * idx]

if dataset == "fashion":
    ## fashion: directly index into the images dataset
    for idx in sort_idx[:10]:
        data_loader.dataset.dataset[idx]
