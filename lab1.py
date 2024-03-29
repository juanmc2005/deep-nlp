import re
import numpy as np
import torch as th
import torch.optim as op
import torch.autograd as ag
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True, help='The number of epochs to run the model')
args = parser.parse_args()


def clean_str(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if tolower:
        string = string.lower()
    return string.strip()


# reads the content of the file passed as an argument.
# if limit > 0, this function will return only the first "limit" sentences in the file.
def loadTexts(filename, limit=-1):
    f = open(filename)
    dataset = []
    line = f.readline()
    cpt = 1
    skip = 0
    while line:
        cleanline = clean_str(f.readline()).split()
        if cleanline:
            dataset.append(cleanline)
        else:
            line = f.readline()
            skip += 1
            continue
        if limit > 0 and cpt >= limit:
            break
        line = f.readline()
        cpt += 1

    f.close()
    print("Load ", cpt, " lines from ", filename, " / ", skip, " lines discarded")
    return dataset


LIM = 5000
txtfile = 'imdb/imdb.pos'  # path of the file containing positive reviews
postxt = loadTexts(txtfile, limit=LIM)

txtfile = 'imdb/imdb.neg'  # path of the file containing negative reviews
negtxt = loadTexts(txtfile, limit=LIM)

x, y = [], []
for sent in postxt:
    x.append(sent)
    y.append(1)
for sent in negtxt:
    x.append(sent)
    y.append(0)

x_train_dev, x_test, y_train_dev, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=10)
x_train, x_dev, y_train, y_dev = train_test_split(x_train_dev, y_train_dev,
                                                  test_size=len(y_test)/len(y_train_dev),
                                                  stratify=y_train_dev, random_state=10)

words = set()
for sent in x_train:
    words.update(sent)

w2id = dict()
id2w = dict()
for i, w in enumerate(words):
    w2id[w] = i
    id2w[i] = w

x_train_i, x_dev_i, x_test_i = [], [], []
for sent in x_train:
    sent_i = th.tensor([w2id[w] for w in sent], requires_grad=False).long()
    x_train_i.append(sent_i)
for sent in x_dev:
    sent_i = th.tensor([w2id[w] for w in sent if w in w2id], requires_grad=False).long()
    x_dev_i.append(sent_i)
for sent in x_test:
    sent_i = th.tensor([w2id[w] for w in sent if w in w2id], requires_grad=False).long()
    x_test_i.append(sent_i)

y_train = th.tensor(y_train, requires_grad=False).float()
y_dev = th.tensor(y_dev, requires_grad=False).float()
# y_test = [th.tensor(label, requires_grad=False).long() for label in y_test]


class CBOW_classifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, nclass):
        super(CBOW_classifier, self).__init__()
        self.emb_table = th.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 50),
            nn.Linear(50, nclass)
        )

    def forward(self, inputs):
        embs = []
        for words in inputs:
            x = self.emb_table(words)
            embs.append(th.sum(x, dim=0))
        embs = th.stack(embs)
        return th.tanh(self.net(embs)).squeeze(1)


model = CBOW_classifier(vocab_size=len(words), embedding_dim=200, nclass=1)
loss_fn = nn.BCEWithLogitsLoss()
optim = op.Adam(model.parameters(), lr=0.01)

for epoch in range(1, args.epochs + 1):

    print(f"[Epoch {epoch}]")

    model.train()

    logits = model(x_train_i)

    loss = loss_fn(logits, y_train)

    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f"[mean loss = {loss.item()}]")

    model.eval()

    with th.no_grad():
        dev_logits = model(x_dev_i)
        pred = (dev_logits > 0).float()
        correct = (pred == y_dev).sum().item()
        total = y_dev.size(0)
        print(f"[dev accuracy = {correct / total}]")

sent_p = ['Worst', 'film', 'I', 'ever', 'watched']
sent_n = ['Best', 'film', 'I', 'ever', 'watched']
sent_pi = th.tensor([w2id[w] for w in sent_p if w in w2id], requires_grad=False).long()
sent_ni = th.tensor([w2id[w] for w in sent_n if w in w2id], requires_grad=False).long()

print(f"{' '.join(sent_p)}: {(model([sent_pi]) > 0).float().item()}")
print(f"{' '.join(sent_n)}: {(model([sent_ni]) > 0).float().item()}")
