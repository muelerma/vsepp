from vocab import Vocabulary
import evaluation

## COCO
evaluation.evalrank("./runs/coco_vse++/model_best.pth.tar", data_path="./data", split="test", vocab_path="./vocab", on_gpu=True)
## t2i top1: 33.8

## f30k
evaluation.evalrank("./runs/f30k_vse++/model_best.pth.tar", data_path="./data", split="test", vocab_path="./vocab", on_gpu=True)
## t2i top1: 23.1

## f8k -> doesnt work with pre-trained models...
evaluation.evalrank("./runs/f30k_vse++/model_best.pth.tar", 
                    data_path="./data", data_name="f8k_precomp", split="test", vocab_path="./vocab", on_gpu=True)

## trained
evaluation.evalrank("./runs/runX/model_best.pth.tar", 
                    data_path="./data", data_name="f8k_precomp", split="test", vocab_path="./vocab", on_gpu=True)
## t2i top1: 14.8