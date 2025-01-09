# models.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformer import Transformer
import time

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, vocab_index):
        super(NeuralLanguageModel, self).__init__()
        self.vocab_index = vocab_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_positions = num_positions
        self.model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers).to(self.device)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_next_char_log_probs(self, context):
        self.model.eval()
        with torch.no_grad():
            context_indices = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0).to(self.device)
            if context_indices.size(1) == 0:
                context_indices = torch.tensor([[self.vocab_index.index_of(' ')]], device=self.device)
            context_indices = context_indices[:, -self.num_positions:]  # Truncate if too long
            
            log_probs, _ = self.model(context_indices)  # Unpack the tuple here
            return log_probs[0, -1].cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        with torch.no_grad():
            context_indices = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0).to(self.device)
            total_log_prob = 0.0
            for char in next_chars:
                if context_indices.size(1) == 0:
                    context_indices = torch.tensor([[self.vocab_index.index_of(' ')]], device=self.device)
                context_indices = context_indices[:, -self.num_positions:]  # Truncate if too long
                
                log_probs, _ = self.model(context_indices)  # Unpack the tuple here
                char_idx = self.vocab_index.index_of(char)
                total_log_prob += log_probs[0, -1, char_idx].item()
                context_indices = torch.cat([context_indices, torch.tensor([[char_idx]], device=self.device)], dim=1)
        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)
    num_positions = 256  # Increased context window
    d_model = 512  # Increased model capacity
    d_internal = 2048  # Increased internal dimension
    num_classes = vocab_size
    num_layers = 6  # Increased number of layers
    batch_size = 64  # Increased batch size
    num_epochs = 30  # Increased number of epochs
    lr = 0.0005  # Slightly increased learning rate
    warmup_steps = 4000
    
    model = NeuralLanguageModel(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, vocab_index)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()
    
    def lr_lambda(step):
        return min((step + 1) ** -0.5, step * (warmup_steps ** -1.5))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def prepare_data(text, chunk_size=257):  # Increased chunk size
        data = [vocab_index.index_of(c) for c in ' ' + text]
        return [data[i:i+chunk_size] for i in range(0, len(data)-chunk_size, chunk_size)]
    
    train_data = prepare_data(train_text)
    dev_data = prepare_data(dev_text)
    
    best_perplexity = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0
        step_count = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            inputs = torch.tensor(batch, dtype=torch.long).to(model.device)
            targets = inputs[:, 1:].contiguous()
            
            optimizer.zero_grad()
            
            log_probs, _ = model.model(inputs[:, :-1])
            loss = criterion(log_probs.view(-1, vocab_size), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Added gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            step_count += 1
        
        model.model.eval()
        dev_loss = 0
        with torch.no_grad():
            for batch in dev_data:
                inputs = torch.tensor(batch, dtype=torch.long).unsqueeze(0).to(model.device)
                targets = inputs[:, 1:].contiguous()
                log_probs, _ = model.model(inputs[:, :-1])
                dev_loss += criterion(log_probs.view(-1, vocab_size), targets.view(-1)).item()
        
        dev_perplexity = math.exp(dev_loss / len(dev_data))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_data):.4f}, Dev Perplexity: {dev_perplexity:.4f}")

        if dev_perplexity < best_perplexity:
            best_perplexity = dev_perplexity
            torch.save(model.state_dict(), 'best_model.pt')

        if time.time() - start_time > 540:  # 9 minutes
            print("Time limit reached")
            break

    return model
