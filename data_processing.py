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

data = ["I like coffee .", "I like tea .", "You like tea .", "You do not like coffee ."]
tokenizer = Tokenizer()

train_ids = []
for s in data:
    train_ids.append(tokenizer.encode(s))

x = [d[:-1] for d in train_ids]
y = [d[1:] for d in train_ids]

print("Our training data x:", x)
print("Our training data y:", y)

