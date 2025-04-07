
import torch 
import torch.nn  as nn 
import torch.nn.functional as F
import math 
from dataclasses import dataclass

# Attention module
class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.fq = nn.Linear(embed_dim, embed_dim)
        self.fk = nn.Linear(embed_dim, embed_dim)
        self.fv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.fq(x)
        k = self.fk(x)
        v = self.fv(x)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
        
        mask = torch.full((q.size(1), q.size(1)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        scores = F.softmax(scores, dim=-1)
        
        return scores @ v

# MLP module
class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(embed_dim, mlp_dim)
        self.fo = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        return self.fo(F.relu(self.fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.self_attn = Attention(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim)
    def forward(self, x):
        h = x + self.self_attn(x) 
        out = h + self.mlp(h) 
        return out


@dataclass
class SimpleLLMConfig:
    vocab_size: int = 8
    embed_dim: int = 2
    n_layers: int = 2
    mlp_dim: int = 6

# Main neural network class integrating Tokenizer, Attention, and MLP
class SimpleLLM(nn.Module):
    def __init__(self, config):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(config.embed_dim, config.mlp_dim))

        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)
        self.output_proj.weight.data = self.embedding.weight

    def forward(self, x , y = None, last_hidden_state = False):
        # Tokenize and embed input
        h = self.embedding(x)

        for layer in self.layers:
            h =  layer(h)
        
        logits = self.output_proj(h)
        if last_hidden_state:
            return h
        if y is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            return logits, loss


if __name__ =='__main__':
    model = SimpleLLM(SimpleLLMConfig)
    print(model)