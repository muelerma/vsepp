"""
Create dataset used for VSE++ based on this Kaggle dataset:
https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset

Data to use for captions:

productDisplayName
brandName
gender
baseColour
usage
displayCategories (split by comma?)
subCategory > typeName
articleType > typeName
articleAttributes (list)

"""
import json
from random import random, choice, sample

## look at first example
with open("10000.json", "rt") as f:
    j = json.load(f)

print(j)

j_dat = j["data"]

j_dat["productDisplayName"]
j_dat["brandName"]
j_dat["gender"]
j_dat["baseColour"]
j_dat["usage"]
j_cats = j_dat["displayCategories"].split(","); j_cats
j_dat["subCategory"]["typeName"]
j_dat["articleType"]["typeName"]
[a for k, a in j_dat["articleAttributes"].items()]


def create_caption(j):

    j_dat = j["data"]

    ## 1) simply use displayName
    cap1 = j_dat["productDisplayName"]

    ## 2) [baseColour] + brandName + gender + articleType + [articleAttributes]
    cap2 = ""
    if random() > 0.5:
        cap2 += j_dat["baseColour"]

    cap2 = " ".join([cap2, j_dat["brandName"], j_dat["gender"], j_dat["articleType"]["typeName"]])

    if random() > 0.5:
        article_attr = [a for k, a in j_dat["articleAttributes"].items() if len(a) < 10]
        if article_attr:
            cap2 += " "
            cap2 += choice([a for k, a in j_dat["articleAttributes"].items() if len(a) < 10])

    ## 3) [baseColour] + usage + articleType + [articleAttributes]
    cap3 = ""
    if random() > 0.5:
        cap3 += j_dat["baseColour"]

    cap3 = " ".join([cap3, j_dat["usage"], j_dat["articleType"]["typeName"]])

    if random() > 0.5:
        article_attr = [a for k, a in j_dat["articleAttributes"].items() if len(a) < 10]
        if article_attr:
            cap3 += " "
            cap3 += choice([a for k, a in j_dat["articleAttributes"].items() if len(a) < 10])

    return cap1, cap2, cap3



## vsepp w/o pre-computed expects captions in json
with open("data/f8k/dataset_flickr8k.json", "rt") as f:
    flickr_caps = json.load(f)

[print(k) for k,v in flickr_caps.items()]

len(flickr_caps["images"])
flickr_caps["dataset"]

flickr_caps["images"][0]
flickr_caps["images"][1]
"""
image example:
{'sentids': [0, 1, 2, 3, 4], 'imgid': 0, 'sentences': [{'tokens': ['a', 'black', 'dog', 'is', 'running', 'after', 'a', 'white', 'dog', 'in', 'the', 'snow'], 'raw': 'A black dog is running after a white dog in the snow .', 'imgid': 0, 'sentid': 0}, {'tokens': ['black', 'dog', 'chasing', 'brown', 'dog', 'through', 'snow'], 'raw': 'Black dog chasing brown dog through snow', 'imgid': 0, 'sentid': 1}, {'tokens': ['two', 'dogs', 'chase', 'each', 'other', 'across', 'the', 'snowy', 'ground'], 'raw': 'Two dogs chase each other across the snowy ground .', 'imgid': 0, 'sentid': 2}, {'tokens': ['two', 'dogs', 'play', 'together', 'in', 'the', 'snow'], 'raw': 'Two dogs play together in the snow .', 'imgid': 0, 'sentid': 3}, {'tokens': ['two', 'dogs', 'running', 'through', 'a', 'low', 'lying', 'body', 'of', 'water'], 'raw': 'Two dogs running through a low lying body of water .', 'imgid': 0, 'sentid': 4}], 'split': 'train', 'filename': '2513260012_03d33305cf.jpg'}

sentids: []
imgid: int
sentences: []
    {tokens: [], raw: str, imgid: int, sentid: 0}
split: str
filename: str
"""

# Modify dataset class in data and user simpler json (see below)


## loop through all style jsons and create captions
from pathlib import Path

data = Path("./styles")
files = data.glob("*")
file_names = [int(f.stem) for f in files]
max(file_names); min(file_names)

img_files = Path("data/fashion/images").glob("*")
img_file_names = [int(f.stem) for f in img_files]
max(img_file_names); min(img_file_names)
## 6 image files less than styles

## create image:captions dict

img_files = Path("data/fashion/images").glob("*")

captions = {}
captions_dev = {}
captions_test = {}
for i, img_file in enumerate(img_files):

    j_file = img_file.stem + ".json"

    try:
        with open("styles/" + j_file, "rt") as f:
            j = json.load(f)
    except:
        print("No style found for image: ", img_file.stem)
    
    c1, c2, c3 = create_caption(j)

    if i % 5 == 1:
        captions_dev[img_file.name] = [c1, c2, c3]
    elif i % 5 == 0:
        captions_test[img_file.name] = [c1, c2, c3]
    else:
        captions[img_file.name] = [c1, c2, c3]


with open("data/fashion/train_caps.json", "wt") as f:
    # f.writelines([c+"\n" for c in captions])
    json.dump(captions, f)

with open("data/fashion/dev_caps.json", "wt") as f:
    # f.writelines([c+"\n" for c in captions_dev])
    json.dump(captions_dev, f)

with open("data/fashion/test_caps.json", "wt") as f:
    # f.writelines([c+"\n" for c in captions_test])
    json.dump(captions_test, f)

## create smaller versions of train and dev
keys = sample(list(captions), len(captions) // 5 )
captions_train_small = {k:captions[k] for k in keys}
with open("data/fashion/train_caps_small.json", "wt") as f:
    json.dump(captions_train_small, f)

keys = sample(list(captions_dev), len(captions_dev) // 5 )
captions_dev_small = {k:captions_dev[k] for k in keys}
with open("data/fashion/dev_caps_small.json", "wt") as f:
    json.dump(captions_dev_small, f)

    
## create vocab
import vocab

vocab.main("./data", "fashion")
## for this to work either write caps as newline seperated txt or modify vocab.from_txt to handle json


## TODO: create image embeddings from pre-trained imagenet model for faster training



