"""
Modified attention extraction for interpretability.
This creates a wrapper around the model to capture attention weights.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model'))

from model import GPT, GPTConfig

class AttentionCapturingGPT(GPT):
    """Modified GPT model that captures attention weights during forward pass."""
    
    def __init__(self, config):
        super().__init__(config)
        self.capture_attention = False
        self.attention_weights = {}
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Store attention weights if capturing
        if self.capture_attention:
            self.attention_weights = {}
        
        for i, block in enumerate(self.transformer.h):
            x = self._forward_block_with_attention(block, x, i)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    
    def _forward_block_with_attention(self, block, x, layer_idx):
        """Forward pass through a block while capturing attention."""
        # LayerNorm 1
        x_normed = block.ln_1(x)
        
        # Attention with capturing
        attn_output = self._forward_attention_with_capture(block.attn, x_normed, layer_idx)
        
        # Residual connection
        x = x + attn_output
        
        # MLP
        x = x + block.mlp(block.ln_2(x))
        
        return x
    
    def _forward_attention_with_capture(self, attn_module, x, layer_idx):
        """Forward attention while capturing weights."""
        B, T, C = x.size()
        
        # Calculate query, key, values
        q, k, v = attn_module.c_attn(x).split(attn_module.n_embd, dim=2)
        k = k.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
        q = q.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
        v = v.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)

        # Compute attention weights
        if attn_module.flash:
            # For flash attention, we can't easily capture weights
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, 
                dropout_p=attn_module.dropout if attn_module.training else 0, 
                is_causal=True
            )
            att = None  # Can't capture with flash attention
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
            
            # Apply causal mask - create one if bias buffer doesn't exist
            if hasattr(attn_module, 'bias') and attn_module.bias is not None:
                att = att.masked_fill(attn_module.bias[:,:,:T,:T] == 0, float('-inf'))
            else:
                # Create causal mask manually
                causal_mask = torch.tril(torch.ones(T, T, device=att.device)).view(1, 1, T, T)
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = attn_module.attn_dropout(att)
            y = att @ v
        
        # Store attention weights if capturing
        if self.capture_attention and att is not None:
            self.attention_weights[f'layer_{layer_idx}'] = att.detach()
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = attn_module.resid_dropout(attn_module.c_proj(y))
        return y

class SimpleAttentionAnalyzer:
    """Simplified attention analyzer that works with any GPT model."""
    
    def __init__(self, model, tokenizer_meta_path: Optional[str] = None):
        self.original_model = model
        self.device = next(model.parameters()).device
        
        # Create attention-capturing version
        self.model = self._create_attention_model(model)
        
        # Setup tokenizer
        if tokenizer_meta_path and os.path.exists(tokenizer_meta_path):
            import pickle
            with open(tokenizer_meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.stoi, self.itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [self.stoi.get(c, self.stoi.get('A', 0)) for c in s]
            self.decode = lambda l: ''.join([self.itos[i] for i in l])
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={""})
            self.decode = lambda l: enc.decode(l)
    
    def _create_attention_model(self, original_model):
        """Create attention-capturing version of the model."""
        # Create new model with same config
        config = original_model.config
        attention_model = AttentionCapturingGPT(config)
        
        # Copy weights
        attention_model.load_state_dict(original_model.state_dict())
        attention_model.to(self.device)
        attention_model.eval()
        
        # Force disable flash attention to capture weights
        for block in attention_model.transformer.h:
            if hasattr(block.attn, 'flash'):
                block.attn.flash = False
        
        return attention_model
    
    def analyze_protein_pair(self, protein1_seq: str, protein2_seq: str, max_length: int = 400) -> Dict:
        """
        Analyze attention patterns for a protein pair.
        
        Args:
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            max_length: Maximum sequence length to process
            
        Returns:
            Dictionary with attention analysis results
        """
        # Format input
        input_text = f"<ps1>,{protein1_seq},<ps2>,{protein2_seq},<"
        
        # Encode and truncate if necessary
        input_ids = self.encode(input_text)
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]  # Keep the end
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Enable attention capturing
        self.model.capture_attention = True
        
        # Forward pass
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
        
        # Get prediction
        probs = F.softmax(logits[:, -1, :], dim=-1)
        
        # Try to get interaction probability
        try:
            token_1_ids = self.encode("1")
            if token_1_ids:
                interaction_prob = probs[0, token_1_ids[0]].item()
            else:
                interaction_prob = None
        except:
            interaction_prob = None
        
        # Parse sequence positions in the tokenized input
        decoded_input = self.decode(input_ids) if hasattr(self, 'decode') else input_text[-len(input_ids):]
        
        # Find approximate positions (this is rough due to tokenization)
        ps1_marker = "<ps1>,"
        ps2_marker = ",<ps2>,"
        
        try:
            ps1_pos = decoded_input.find(ps1_marker)
            ps2_pos = decoded_input.find(ps2_marker)
            
            if ps1_pos >= 0 and ps2_pos >= 0:
                protein1_start = ps1_pos + len(ps1_marker)
                protein1_end = ps2_pos
                protein2_start = ps2_pos + len(ps2_marker)
                protein2_end = len(decoded_input) - 2  # Remove trailing ",<"
                
                # Convert to token positions (approximate)
                protein1_token_start = len(self.encode(decoded_input[:protein1_start]))
                protein1_token_end = len(self.encode(decoded_input[:protein1_end]))
                protein2_token_start = len(self.encode(decoded_input[:protein2_start]))
                protein2_token_end = len(self.encode(decoded_input[:protein2_end]))
            else:
                # Fallback estimates
                protein1_token_start = 5
                protein1_token_end = min(protein1_token_start + len(protein1_seq), len(input_ids) // 2)
                protein2_token_start = protein1_token_end + 5
                protein2_token_end = min(protein2_token_start + len(protein2_seq), len(input_ids) - 5)
        except:
            # Simple fallback
            protein1_token_start = 5
            protein1_token_end = len(input_ids) // 2
            protein2_token_start = len(input_ids) // 2 + 5
            protein2_token_end = len(input_ids) - 5
        
        # Disable attention capturing
        self.model.capture_attention = False
        
        return {
            'attention_weights': self.model.attention_weights.copy(),
            'input_ids': input_ids,
            'interaction_prob': interaction_prob,
            'sequence_info': {
                'protein1_seq': protein1_seq,
                'protein2_seq': protein2_seq,
                'protein1_tokens': (protein1_token_start, protein1_token_end),
                'protein2_tokens': (protein2_token_start, protein2_token_end),
                'total_tokens': len(input_ids),
                'decoded_input': decoded_input
            }
        }
    
    def get_cross_attention(self, attention_data: Dict, layer_idx: int = -1) -> np.ndarray:
        """Extract cross-protein attention matrix."""
        if not attention_data['attention_weights']:
            return np.array([[]])
        
        # Get layer attention
        layer_key = f'layer_{layer_idx}' if layer_idx >= 0 else list(attention_data['attention_weights'].keys())[layer_idx]
        if layer_key not in attention_data['attention_weights']:
            return np.array([[]])
        
        attention = attention_data['attention_weights'][layer_key]  # [batch, heads, seq_len, seq_len]
        
        # Average across heads and remove batch dimension
        attention_avg = attention.mean(dim=1).squeeze(0).cpu().numpy()
        
        # Get protein token ranges
        seq_info = attention_data['sequence_info']
        p1_start, p1_end = seq_info['protein1_tokens']
        p2_start, p2_end = seq_info['protein2_tokens']
        
        # Extract cross-attention
        if p1_end <= attention_avg.shape[0] and p2_end <= attention_avg.shape[1]:
            cross_attention = attention_avg[p1_start:p1_end, p2_start:p2_end]
        else:
            # Fallback to smaller region
            max_p1 = min(p1_end, attention_avg.shape[0])
            max_p2 = min(p2_end, attention_avg.shape[1])
            cross_attention = attention_avg[p1_start:max_p1, p2_start:max_p2]
        
        return cross_attention
    
    def get_top_interactions(self, cross_attention: np.ndarray, protein1_seq: str, 
                           protein2_seq: str, top_k: int = 10) -> List[Dict]:
        """Get top residue interactions based on attention."""
        if cross_attention.size == 0:
            return []
        
        # Find top attention positions
        flat_attention = cross_attention.flatten()
        top_indices = np.argsort(flat_attention)[-top_k:][::-1]
        
        interactions = []
        for idx in top_indices:
            i, j = np.unravel_index(idx, cross_attention.shape)
            
            # Map back to sequence positions (approximate)
            seq1_pos = min(i, len(protein1_seq) - 1)
            seq2_pos = min(j, len(protein2_seq) - 1)
            
            interaction = {
                'protein1_pos': seq1_pos,
                'protein1_residue': protein1_seq[seq1_pos] if seq1_pos < len(protein1_seq) else 'X',
                'protein2_pos': seq2_pos,
                'protein2_residue': protein2_seq[seq2_pos] if seq2_pos < len(protein2_seq) else 'X',
                'attention_score': cross_attention[i, j],
                'token_pos_1': i,
                'token_pos_2': j
            }
            interactions.append(interaction)
        
        return interactions


def main():
    """Test the simplified attention analyzer."""
    print("SimpleAttentionAnalyzer ready for use!")
    
if __name__ == "__main__":
    main()