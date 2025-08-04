from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .llama_moe_mms import LlamaModel, LlamaConfig, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM
from dataclasses import dataclass
from transformers.utils import ModelOutput
@dataclass
class MMoeCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).

        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.

        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    aux_loss_vis: Optional[torch.FloatTensor] = None
    aux_loss_lan: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    vis_router_logits: Optional[Tuple[torch.FloatTensor]] = None
    lan_router_logits: Optional[Tuple[torch.FloatTensor]] = None



class BunnyMMSLlamaMoEConfig(LlamaConfig):
    model_type = "bunny-mm-llama3-moe-s"

    def __init__(self,
                 num_experts_per_tok=2,
                 num_experts = 4,
                 vis_router_aux_loss_coef=0.001,
                 lan_router_aux_loss_coef=0.001,
                 output_vis_router_logits = True,
                 output_lan_router_logits = True,
                 **kwargs):
        self.num_experts_per_tok=num_experts_per_tok
        self.num_experts =  num_experts
        self.vis_router_aux_loss_coef = vis_router_aux_loss_coef
        self.output_vis_router_logits = output_vis_router_logits
        self.lan_router_aux_loss_coef = lan_router_aux_loss_coef
        self.output_lan_router_logits = output_lan_router_logits
        
        super(LlamaConfig , self).__init__(**kwargs)


class BunnyMMSLlamaMoEModel(BunnyMetaModel, LlamaModel):
    config_class = BunnyMMSLlamaMoEConfig

    def __init__(self, config: LlamaConfig):
        super(BunnyMMSLlamaMoEModel, self).__init__(config)


class BunnyMMSLlamaMoEForCausalLM(LlamaForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyMMSLlamaMoEConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = BunnyMMSLlamaMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vis_router_aux_loss_coef = config.vis_router_aux_loss_coef
        self.lan_router_aux_loss_coef = config.lan_router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            token_type_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_vis_router_logits: Optional[bool] = None,
            output_lan_router_logits: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MMoeCausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_vis_router_logits=output_vis_router_logits,
            output_lan_router_logits=output_lan_router_logits,
            return_dict=return_dict,
            cache_position=None
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images

        return _inputs


AutoConfig.register("bunny-mm-llama3-moe-s", BunnyMMSLlamaMoEConfig)
AutoModelForCausalLM.register(BunnyMMSLlamaMoEConfig, BunnyMMSLlamaMoEForCausalLM)
