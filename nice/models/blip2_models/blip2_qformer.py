"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer as _Blip2Qformer
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures

# unregister existing BLIP2 model from registry
if "blip2" in registry.mapping["model_name_mapping"]:
    registry.mapping["model_name_mapping"].pop("blip2", None)
if "blip2_feature_extractor" in registry.mapping["model_name_mapping"]:
    registry.mapping["model_name_mapping"].pop("blip2_feature_extractor", None)


@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(_Blip2Qformer):

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        # image = samples.get("image")
        caption = samples.get("text_input")
        # assert mode is one of "image", "text", "multimodal"
        assert mode in ["image", "text", "multimodal"]

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            pass

        elif mode == "text":
            assert (caption is not None), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len
            ).to(self.device)

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            # text_features = self.text_proj(text_embeds)
            # text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            pass

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
