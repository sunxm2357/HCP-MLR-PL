import torch
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
import numpy as np

tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch

tokenizer("Hello, How are you?")

global_vectors = GloVe(name='840B', dim=300)

classnames = ["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock", "field",
                   "grass", "mobile-home", "pavement", "sand", "sea", "ship", "tanks", "trees", "water"]

embeddings = []
for classname in classnames:
    embedding = global_vectors.get_vecs_by_tokens(tokenizer(classname), lower_case_backup=True)
    embeddings.append(embedding)
embeddings = torch.cat(embeddings, dim=0)
embeddings = embeddings.numpy()
np.save('big_earth.npy', embeddings)
