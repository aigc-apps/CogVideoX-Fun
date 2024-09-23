from typing import Optional
import torch
from flash_attn import flash_attn_func
from einops import rearrange

from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb


class CogVideoXSWAAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, window_size=1024):
        self.window_size = window_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        num_frames: int = None, 
        height: int = None, 
        width: int = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim) # .transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
        
        query = query.transpose(1, 2).to(value)
        key = key.transpose(1, 2).to(value)

        interval = max((query.size(1) - text_seq_length) // (self.window_size - 256), 1)
        cross_key = torch.cat([key[:, :text_seq_length], key[:, text_seq_length::interval]], dim=1)
        cross_val = torch.cat([value[:, :text_seq_length], value[:, text_seq_length::interval]], dim=1)
        cross_hidden_states = flash_attn_func(query, cross_key, cross_val, dropout_p=0.0, causal=False)
        query_txt = query[:, :text_seq_length]
        key_txt = key[:, :text_seq_length]
        value_txt = value[:, :text_seq_length]
        querys = torch.tensor_split(query[:, text_seq_length:], 6, 2)
        keys = torch.tensor_split(key[:, text_seq_length:], 6, 2)
        values = torch.tensor_split(value[:, text_seq_length:], 6, 2)
        new_querys = [querys[0]]
        new_keys = [keys[0]]
        new_values = [values[0]]
        for index, mode in enumerate(["bs (f h w) hn hd -> bs (f w h) hn hd", "bs (f h w) hn hd -> bs (h f w) hn hd", "bs (f h w) hn hd -> bs (h w f) hn hd", 
                                      "bs (f h w) hn hd -> bs (w f h) hn hd", "bs (f h w) hn hd -> bs (w h f) hn hd"]):
            new_querys.append(rearrange(querys[index + 1], mode, f=num_frames, h=height, w=width))
            new_keys.append(rearrange(keys[index + 1], mode, f=num_frames, h=height, w=width))
            new_values.append(rearrange(values[index + 1], mode, f=num_frames, h=height, w=width))
        query = torch.cat([query_txt, torch.cat(new_querys, dim=2)], dim=1)
        key = torch.cat([key_txt, torch.cat(new_keys, dim=2)], dim=1)
        value = torch.cat([value_txt, torch.cat(new_values, dim=2)], dim=1)
        
        hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, causal=False, window_size=(self.window_size, self.window_size))
        hidden_states_txt = hidden_states[:, :text_seq_length]
        hidden_states = torch.tensor_split(hidden_states[:, text_seq_length:], 6, 2)
        new_hidden_states = [hidden_states[0]]
        for index, mode in enumerate(["bs (f w h) hn hd -> bs (f h w) hn hd", "bs (h f w) hn hd -> bs (f h w) hn hd", "bs (h w f) hn hd -> bs (f h w) hn hd", 
                                      "bs (w f h) hn hd -> bs (f h w) hn hd", "bs (w h f) hn hd -> bs (f h w) hn hd"]):
            new_hidden_states.append(rearrange(hidden_states[index + 1], mode, f=num_frames, h=height, w=width))
        hidden_states = torch.cat([hidden_states_txt, torch.cat(new_hidden_states, dim=2)], dim=1) + cross_hidden_states


        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
