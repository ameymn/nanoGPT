import torch

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

device = torch.device("mps")
max_iters = 3000
eval_interval = 300
block_size=8
batch_size = 32

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(len(text))
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}

itos = {i: ch for i, ch in enumerate(chars)}



def encode(s): return [stoi[i] for i in s]


def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long).to(device)

n = int(0.9*(len(data)))
train_data = data[:n]

validation_data = data[n:]

def batch(split):
    data = train_data if split == "trian" else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size=65)
m = m.to(device)

#print(decode(m.generate(torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for step in range(10000):
    xb, yb = batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


print(decode(m.generate(torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=500)[0].tolist()))



#for iter in range(max_iters):

    #if iter%eval_interval == 0:
        #losses = estimate_loss()
        #print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")