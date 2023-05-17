import torch
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_opt import Blip2OPT as _Blip2OPT

# unregister existing BLIP2 model from registry
if "blip2_opt" in registry.mapping["model_name_mapping"]:
    registry.mapping["model_name_mapping"].pop("blip2_opt", None)


@registry.register_model("blip2_opt")
class Blip2OPT(_Blip2OPT):

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        fuse_ret_feature=None,
        ret_k=None
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len
        )
        if fuse_ret_feature is not None:
            assert fuse_ret_feature in ["query_output", "visual_patch"]
            proj_dim = (
                self.visual_encoder.embed_dim
                if fuse_ret_feature == "visual_patch" else self.Qformer.config.hidden_size
            )
            self.ret_proj = nn.Linear(self.Qformer.config.hidden_size, proj_dim)
        self.fuse_ret_feature = fuse_ret_feature
        self.ret_k = ret_k

    def fuse_retrival_samples(self, image_embeds, ret_features):
        # (B, M, L, D) -> (B, L, D) (by avg)
        assert self.ret_k is not None
        ret_features = ret_features[:, :self.ret_k].to(image_embeds.device).mean(dim=1)
        # project 'ret_features' to vision space and concat with image embeddings
        return torch.cat([image_embeds, self.ret_proj(ret_features)], dim=1)

    def _forward(self, samples):  # normal forward mtd in training
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        if "ret_features" in samples and self.fuse_ret_feature == "visual_patch":
            image_embeds = self.fuse_retrival_samples(image_embeds, samples["ret_features"])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ).last_hidden_state
        if "ret_features" in samples and self.fuse_ret_feature == "query_output":
            query_output = self.fuse_retrival_samples(query_output, samples["ret_features"])

        inputs_opt = self.opt_proj(query_output)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, :self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100))
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    def forward(self, samples, extract_feats=False, agg_mode="avg", normalize=True):
        if extract_feats:
            return self.extract_visual_features(samples, agg_mode=agg_mode, normalize=normalize)
        else:
            return self._forward(samples)

    def _extract_visual_features(self, samples):
        image = samples["image"]
        with torch.cuda.amp.autocast(dtype=torch.float16):
            image_embeds = self.visual_encoder(image)
        return image_embeds.float()

    @torch.no_grad()
    def extract_visual_features(self, samples, agg_mode="avg", normalize=True):
        # this is for retrieval comparison of visual representation
        image_embeds = self._extract_visual_features(samples)
        if agg_mode == "class_token_only":
            image_embeds = image_embeds[:, 0]  # (B, L, D) -> (B, D)
        elif agg_mode == "avg":
            image_embeds = image_embeds.mean(dim=1)
        else:
            pass
        if normalize:
            image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            if "ret_features" in samples and self.fuse_ret_feature == "visual_patch":
                image_embeds = self.fuse_retrival_samples(image_embeds, samples["ret_features"])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            ).last_hidden_state

            if "ret_features" in samples and self.fuse_ret_feature == "query_output":
                query_output = self.fuse_retrival_samples(query_output, samples["ret_features"])

            inputs_opt = self.opt_proj(query_output)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(image.device)
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")  # 32
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)  # input to OPT

        fuse_ret_feature = cfg.get("fuse_ret_feature", None)
        ret_k = cfg.get("ret_k", None)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            fuse_ret_feature=fuse_ret_feature,
            ret_k=ret_k
        )
        model.load_checkpoint_from_config(cfg)

        return model
