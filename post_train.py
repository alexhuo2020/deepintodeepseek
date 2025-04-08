

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SimpleLLM, SimpleLLMConfig
from dataclasses import dataclass
from transformer import Tokenizer
from generate import generate
@dataclass
class TrainConfig:
    lr: float = 1e-3
    max_epochs: int = 2000

class Tokenizer:
    def __init__(self):
        self.word_to_id = {"I":0, "You":1, "like": 2, "do": 3, "not": 4, "coffee":5,"tea":6,".": 7, "human": 8, "system": 9}
        self.id_to_word = {v:k for k,v in self.word_to_id.items()}
    def encode(self, x):
        return [self.word_to_id[c] for c in x.split()]
    def decode(self, x):
        return ' '.join([self.id_to_word[c.item()] for c in x])

# resize the tokenization
def resize_token_embeddings(embed: nn.Embedding, num_new_tokens: int) -> nn.Embedding:
    old_vocab_size, embedding_dim = embed.weight.shape
    new_vocab_size = old_vocab_size + num_new_tokens

    # Create new embedding layer
    new_embed = nn.Embedding(new_vocab_size, embedding_dim)

    # Copy weights from the old embedding
    new_embed.weight.data[:old_vocab_size] = embed.weight.data

    # Initialize the new embeddings (e.g., normal)
    nn.init.normal_(new_embed.weight.data[old_vocab_size:])

    return new_embed



if __name__ == '__main__':
    # preparing data
    data = ["human do I like coffee . system do", "human do You like tea . system do", "human do I like tea . system do", "human do You like coffee . system not","human do I like coffee . system like"]
    tokenizer = Tokenizer()

    train_ids = []
    for s in data:
        train_ids.append(tokenizer.encode(s))

    x = [torch.tensor(d[:-1]) for d in train_ids]
    y = [torch.tensor(d[1:]) for d in train_ids]
    train_ids = list(zip(x,y))

    # define the model
    model = SimpleLLM(SimpleLLMConfig)
    model.load_state_dict(torch.load("model.pt", weights_only=True))

    model.embedding = resize_token_embeddings(model.embedding, 2)
    model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
    model.output_proj.weight = model.embedding.weight  # re-tie after replacing the embedding


    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.lr)

    # training
    losses = []
    for epoch in range(3000):
        for x, y in train_ids:
            _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
    torch.save(model.state_dict(), "model_sft.pt")

    import matplotlib.pyplot as plt 
    plt.plot(losses)
    plt.show()
    plt.savefig("loss.png")
    # evaluation
    X = ["human do I like coffee . system ", "human do You like coffee . system"]
    for x in X:
        x = torch.tensor([tokenizer.encode(x)])

        logits = model(x)
        print(F.softmax(logits, -1))
        for _ in range(10):
            y = generate(model, x, 1)
            for yy in y:
                print(tokenizer.decode(yy))


