import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import math 
import torch.nn.functional as F 

class Tokenizer:
    def __init__(self):
        self.word_to_id = {"I":0, "You":1, "like": 2, "do": 3, "not": 4, "coffee":5,"tea":6,".": 7}
        self.id_to_word = {v:k for k,v in self.word_to_id.items()}
    def encode(self, x):
        return [self.word_to_id[c] for c in x.split()]
    def decode(self, x):
        return ' '.join([self.id_to_word[c.item()] for c in x])

if __name__ == '__main__':
    x1 = "I like"
    tokenizer = Tokenizer()
    x2 = "You do not like coffee ."
    print("The tokenized result for string: %s is "%x1, tokenizer.encode(x1))
    print("The tokenized result for string: %s is "%x2, tokenizer.encode(x2))

    vocab = tokenizer.word_to_id.keys()
    vocab_size = 8
    embed_dim = 2
    mlp_dim = 32

    # Embedding Layer
    fe = nn.Embedding(vocab_size, embed_dim)
    fig, ax = plt.subplots()
    import numpy as np 
    x = torch.arange(vocab_size)
    xe = fe(x)
    print("embedding of the vocabs: ", xe)

    x1e = fe(torch.tensor([tokenizer.encode(x1)]))
    print("embedding for %s is:"%x1, x1e)

    fig, ax = plt.subplots()
    import numpy as np 

    ax.scatter(xe.detach()[:,0], xe.detach()[:,1], s=50, alpha=0.5)
    for i, txt in enumerate(vocab):
        ax.annotate(txt, (xe[i,0],xe[i,1]))


    # Linear transformation
    l = nn.Linear(embed_dim, embed_dim)
    x1l = l(x1e)
    ax.scatter(x1l.detach()[0][:,0], x1l.detach()[0][:,1], s=50, alpha=0.5)
    for i, txt in enumerate(x1.split(" ")):
        print(i, txt)
        ax.annotate(txt, (x1l[0][i,0],x1l[0][i,1]))
    plt.show()


    # Attention
    seqlen = len(x1.split(" "))
    fv = nn.Linear(embed_dim, embed_dim)
    fq = nn.Linear(embed_dim, embed_dim)
    fk = nn.Linear(embed_dim, embed_dim)
    q = fq(x1e)
    k = fk(x1e)
    v = fv(x1e)
    scores = q @ k.transpose(-1,-2) / math.sqrt(embed_dim)
    print("attention coeff. without softmax and mask: ", scores)
    mask = torch.full((seqlen, seqlen), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    scores = scores + mask
    print("attention coeff. without softmax after masking: ", scores)
    scores = F.softmax(scores, dim=-1)
    print("attention coeff. after softmax and masking: ", scores)

    attn_output = scores @ v
    print("attention output: ", attn_output)
    ax.scatter(attn_output.detach()[0][:,0], attn_output.detach()[0][:,1], s=50, alpha=0.5)
    for i, txt in enumerate(x1.split(" ")):
        print(i, txt)
        ax.annotate(txt, (attn_output[0][i,0],attn_output[0][i,1]))
    plt.show()

    # MLP
    fc = nn.Linear(embed_dim, mlp_dim)
    fo = nn.Linear(mlp_dim, embed_dim)
    mlp_output = fo(F.relu(fc(attn_output)))
    # output projection
    f_proj = nn.Linear(embed_dim, vocab_size)
    f_proj.weight.data = fe.weight
    logits = f_proj(mlp_output)
    print("logits are: ", logits)
    probs = F.softmax(logits, -1)
    print("probabilities are: ", probs)

    # inference
    next_token_probs = probs[:,-1]
    import seaborn as sns


    plt.figure()
    print(vocab)
    print(next_token_probs.shape)
    plt.bar(vocab, next_token_probs[0].detach())
    plt.show()
    next_token = torch.multinomial(next_token_probs, num_samples=1)
    print(next_token)
    print(tokenizer.decode(next_token))



        
