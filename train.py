import json
import os

import torch

from chatspace.data.vocab import Vocab
from chatspace.model import ChatSpaceModel
from chatspace.resource import CONFIG_PATH, JIT_MODEL_PATH, MODEL_DICT_PATH, VOCAB_PATH
from chatspace.train.trainer import ChatSpaceTrainer

CORPUS_PATH = os.environ["CORPUS_PATH"]

with open(CONFIG_PATH) as f:
    config = json.load(f)

vocab = Vocab.load(VOCAB_PATH, with_forward_special_tokens=True)
config["vocab_size"] = len(vocab)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ChatSpaceModel(config).to(device)

trainer = ChatSpaceTrainer(config, model, vocab, device, train_corpus_path=CORPUS_PATH)
trainer.train(epochs=config['epochs'], batch_size=config['batch_size'])

# trainer.load_model(MODEL_PATH)
trainer.save_model(JIT_MODEL_PATH, as_jit=True)
trainer.save_model(MODEL_DICT_PATH, as_jit=False)
