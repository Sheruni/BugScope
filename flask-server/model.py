# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
from custom_cross_attention import CustomCrossAttention
class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.cross_attention = CustomCrossAttention(embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads)
        self.fc = nn.Linear(self.config.hidden_size*2, self.config.hidden_size)
        self.gelu = nn.GELU()
    
    def forward(self, code_inputs, nl_inputs, return_vec=False): 
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        
        # Encoder
        model_outputs = self.encoder(inputs, attention_mask=inputs.ne(1))
        outputs = model_outputs[1]
        
        # outputs = model_outputs.logits
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        
        # Cross-Attention
        # Add sequence dimension to code_vec and nl_vec
        code_vec_unsqueezed = code_vec.unsqueeze(0)
        nl_vec_unsqueezed = nl_vec.unsqueeze(0)

        # Cross-Attention
        cross_attn_output, _ = self.cross_attention(nl_vec_unsqueezed, code_vec_unsqueezed, code_vec_unsqueezed)
        cross_attn_output = cross_attn_output.squeeze(0)
        
        # Concatenate code vectors and cross-attention output
        code_vec = torch.cat((code_vec, cross_attn_output), dim=-1)

        # Add fully connected layer
        code_vec = self.fc(code_vec)
        code_vec = self.gelu(code_vec)
        
        if return_vec:
            return None, code_vec, nl_vec
        
        # Compute similarity scores
        scores = (nl_vec[:,None,:] * code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss, code_vec, nl_vec
    
    
    
    
    
    