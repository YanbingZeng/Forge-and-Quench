import os
import json
import torch
import torch.nn as nn
from typing import Optional
from safetensors.torch import load_file

from diffusers import ConfigMixin, ModelMixin
from diffusers.models.normalization import RMSNorm
from transformers import Qwen2_5_VLForConditionalGeneration
from .mmdit import FluxTransformer2DModel

class ForgeModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    def __init__(
            self,
            qwen25_vl_path,
            forge_model_path
    ):
        super().__init__()
        config_file = os.path.join(forge_model_path, 'config.json')
        config = json.load(open(config_file))
        self.num_querys = config["num_querys"]
        self.in_channels = config["in_channels"]

        self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen25_vl_path)
        self.dit = FluxTransformer2DModel(
            in_channels=config["in_channels"],
            num_layers=config["num_layers"],
            num_single_layers=config["num_single_layers"],
            joint_attention_dim=config["joint_attention_dim"],
        )

        self.gen_mlp_adapter = torch.nn.Sequential(
            nn.Linear(self.mllm.config.hidden_size, config["joint_attention_dim"]),
            nn.GELU(),
            nn.Linear(config["joint_attention_dim"], config["joint_attention_dim"])
        )
        
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_querys, self.mllm.config.hidden_size))
        
        model_state = load_file(os.path.join(forge_model_path, 'forge_model.safetensors'))
        m, u = self.load_state_dict(model_state, strict=False)
        assert len(u) == 0

    @torch.inference_mode
    def forward_prompt_embs(
            self,
            input_embeds,
            attention_mask,
            offset=3,
    ):  
        output = self.mllm(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states[-1]

        prompt_embeds = hidden_states[:, -(offset+self.num_querys):-offset, :]
        prompt_embeds = self.gen_mlp_adapter(prompt_embeds)
        
        return prompt_embeds


class FaQAttnProcessor(torch.nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""
    
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        
        self.faq_adapter_to_k = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=True)
        self.faq_adapter_to_v = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=True)
        self.faq_adapter_norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        adapter_feature: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if adapter_feature is not None:
            # `faq-adapter` projections
            faq_hidden_states = adapter_feature
            faq_hidden_states_key_proj = self.faq_adapter_to_k(faq_hidden_states)
            faq_hidden_states_value_proj = self.faq_adapter_to_v(faq_hidden_states)

            faq_hidden_states_key_proj = faq_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            faq_hidden_states_value_proj = faq_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            faq_hidden_states_key_proj = self.faq_adapter_norm_added_k(faq_hidden_states_key_proj)

            faq_hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query, 
                faq_hidden_states_key_proj, 
                faq_hidden_states_value_proj, 
                dropout_p=0.0, 
                is_causal=False
            )

            faq_hidden_states = faq_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            faq_hidden_states = faq_hidden_states.to(query.dtype)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2) # (512+3840,128)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, 
            key, 
            value, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        if encoder_hidden_states is not None:

            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            if adapter_feature is not None:
                hidden_states = hidden_states + self.scale * faq_hidden_states
                        
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            return hidden_states, encoder_hidden_states
        else:
            if adapter_feature is not None:
                hidden_states = hidden_states + self.scale * faq_hidden_states
            
            return hidden_states
