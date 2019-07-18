import json
import os

import torch

from chatspace.data.vocab import Vocab
from chatspace.model import ChatSpaceModel
from chatspace.resource import CONFIG_PATH, VOCAB_PATH
from chatspace.train.trainer import ChatSpaceTrainer

CORPUS_PATH = os.environ["CORPUS_PATH"]

with open(CONFIG_PATH) as f:
    config = json.load(f)

vocab = Vocab.load(VOCAB_PATH, with_forward_special_tokens=True)
config["vocab_size"] = len(vocab)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ChatSpaceModel(config).to(device)

trainer = ChatSpaceTrainer(config, model, vocab, device, train_corpus_path=CORPUS_PATH)
trainer.train(batch_size=512)
