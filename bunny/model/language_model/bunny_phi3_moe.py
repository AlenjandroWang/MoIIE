from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .phi3_moe import Phi3Model, Phi3Config, Phi3ForCausalLM

from transformers.modeling_outputs import  MoeCausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM


class BunnyPhi3MoEConfig(Phi3Config):
    model_type = "bunny-phi3-moe"

    def __init__(self,
                 num_experts_per_tok=2,
                 num_experts = 4,
                 router_aux_loss_coef=0.001,
                 output_router_logits = True,
                 **kwargs):
        self.num_experts_per_tok=num_experts_per_tok
        self.num_experts =  num_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
     
        super(Phi3Config , self).__init__(**kwargs)


class BunnyPhi3MoEModel(BunnyMetaModel, Phi3Model):
    config_class = BunnyPhi3MoEConfig

    def __init__(self, config: Phi3Config):
        super(BunnyPhi3MoEModel, self).__init__(config)


class BunnyPhi3MoEForCausalLM(Phi3ForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyPhi3MoEConfig

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = BunnyPhi3MoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict
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


AutoConfig.register("bunny-phi3-moe", BunnyPhi3MoEConfig)
AutoModelForCausalLM.register(BunnyPhi3MoEConfig, BunnyPhi3MoEForCausalLM)
