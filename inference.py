import torch

from model import VSE
from vocab import Vocabulary
from data import get_test_loader, get_loaders
from evaluation import encode_data

import numpy as np
import nltk
import pickle
import os
import argparse


def load_model(checkpoint, dataset):
    """load fine-tuned model and vocabulary"""
    opt = checkpoint['opt']

    # load vocabulary used by the model
    with open(f"./vocab/{dataset}_vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    assert opt.vocab_size == len(vocab)

    model = VSE(opt)
    model.load_state_dict(checkpoint['model'])

    return model, vocab


def get_data_loader(t, vocab, opt):
    print('Loading dataset: ', t)
    if t == "test":
        data_loader = get_test_loader("test", opt.data_name, vocab, opt.crop_size,
                                    opt.batch_size, opt.workers, opt)
    elif t == "dev": 
        _, data_loader = get_loaders(opt.data_name, vocab, opt.crop_size,
                                    opt.batch_size, opt.workers, opt)
    else:
        data_loader, _ = get_loaders(opt.data_name, vocab, opt.crop_size,
                                    opt.batch_size, opt.workers, opt)
    return data_loader


def create_embds(checkpoint, model, vocab, on_gpu):
    """create embeddings for all images"""
    opt = checkpoint['opt']

    for t in ["train", "dev", "test"]:
        
        data_loader = get_data_loader(t, vocab, opt)

        img_embs, cap_embs = encode_data(model, data_loader, on_gpu=on_gpu)

        ## keep only one emb per image
        c_per_i = data_loader.dataset.caps_per_img
        mask = np.array([i for i in range(img_embs.shape[0]) if i % c_per_i == 0])
        img_embs = img_embs[mask]

        assert img_embs[0].shape[0] == cap_embs[0].shape[0]

        ## save as numpy arrays
        np.save(f"./data/fashion/{t}_ims.npy", img_embs)


def load_embs(embs_path, split):
    return np.load(os.path.join(embs_path, f"{split}_ims.npy"))


def query_embd(query, model, vocab, on_gpu):
    """from PrecomputeDataset: Convert caption (string) to word ids."""
    tokens = nltk.tokenize.word_tokenize(
        str(query).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.tensor(caption, dtype=torch.long)
    if on_gpu: 
        target = target.cuda()

    txt_emb = model.txt_enc(target.unsqueeze(0), torch.tensor([len(target),]))
    return txt_emb[0].detach().cpu().numpy()


def top_k_img(txt_emb, img_embs, k, data_loader):
    """search k most similar img embeddings"""
    scores = np.dot(img_embs, txt_emb)

    sort_idx = np.argsort(scores)[::-1]
    scores[sort_idx[:k]]

    imgs = []
    for idx in sort_idx[:k]:
        imgs.append(data_loader.dataset.dataset[idx])
    return imgs


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", help="Query to score images against")
    parser.add_argument("-k", help="How many images to return", default=16)
    parser.add_argument("--create_embeddings", "-c", action="store_true", help="Create image embeddings for train, dev, test set")
    parser.add_argument("--split", "-s", help="Score query against train/dev/test set", default="test")
    parser.add_argument("--dataset", "-d", help="Which dataset to use fashion/flickr/coco", default="fashion")
    parser.add_argument("--model", "-m", help="Path to model tar", default="./runs/runX/model_best.pth.tar")
    parser.add_argument("--vocab", "-v", help="Path to vocabulary", default="./vocab")
    parser.add_argument("--embeddings", "-e", help="Path to numpy embeddings", default="./data/fashion/")
    parser.add_argument("--gpu", "-g", action="store_true")

    args = parser.parse_args()

    if not args.gpu:
        raise Exception("Need to set --gpu flag: Inference is currently only working with a cuda capable GPU")
    if args.gpu: device = "cuda"
    # else: device = "cpu"

    checkpoint = torch.load(args.model, map_location=torch.device(device))

    if args.create_embeddings:
        model, vocab = load_model(checkpoint, args.dataset)
        create_embds(checkpoint, model, vocab, args.gpu)
    else:
        model, vocab = load_model(checkpoint, args.dataset)
        
        img_embs = load_embs(args.embeddings, args.split)
        txt_emb = query_embd(args.query, model, vocab, args.gpu)

        data_loader = get_data_loader(args.split, vocab, checkpoint["opt"])
        imgs = top_k_img(txt_emb, img_embs, args.k, data_loader)

        print("Top k images for your query: ", imgs)
