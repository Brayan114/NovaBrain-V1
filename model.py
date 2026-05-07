import torch
import torch.nn as nn
import torch.nn.functional as F

class OrbitTransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        # 1. The Hydra (Multi-Head Attention)
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        
        # 2. The Muscle (Feed-Forward Network)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(), # The non-linear curve so the AI can think in complex shapes
            nn.Linear(embed_size * 4, embed_size)
        )
        
        # 3. The Stabilizers (Layer Normalization)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # --- THE HIGHWAY SYSTEM ---
        # Step 1: Hydra processing + The Highway addition
        attention_output, _ = self.attention(x, x, x) 
        x = self.norm1(x + attention_output) 
        
        # Step 2: Muscle processing + The Highway addition
        ffwd_output = self.feed_forward(x)
        x = self.norm2(x + ffwd_output) 
        
        return x

class OrbitLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super().__init__()
        
        # 1. The ID Cards (Turning integers into vectors)
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(1000, embed_size) 
        
        # 2. The Skyscraper Floors (Stacking multiple Transformer Blocks)
        self.blocks = nn.Sequential(*[
            OrbitTransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        
        # 3. Final Stabilization
        self.ln_f = nn.LayerNorm(embed_size)
        
        # 4. The Un-Grinder
        self.language_head = nn.Linear(embed_size, vocab_size)

    def forward(self, input_tokens):
        # Step 1: Generate the GPS coordinates (0, 1, 2, 3...) for the length of the sentence
        positions = torch.arange(input_tokens.shape[1]) 
        
        # Step 1.5: Give tokens their ID cards AND add their GPS stamps!
        x = self.token_embedding_table(input_tokens) + self.position_embedding_table(positions) 
        
        # Step 2: Send data up through all the floors of the Skyscraper
        x = self.blocks(x) 
        
        # Step 3: Final stabilization
        x = self.ln_f(x)
        
        # Step 4: The Un-Grinder predicts the next word scores
        final_predictions = self.language_head(x)
        
        return final_predictions
