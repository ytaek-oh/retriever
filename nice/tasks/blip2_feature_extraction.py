import os

import torch.distributed as dist

import h5py
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("blip2_extract_q_former")
class Blip2ExtractQFormer(BaseTask):

    def __init__(self, feature_mode, target_dir):
        self.feature_mode = feature_mode
        self.target_dir = target_dir

    def valid_step(self, model, samples):
        # model registry: blip2_feature_extractor or blip2
        results = []
        outputs = model.extract_features(samples, mode=self.feature_mode)
        features = (
            outputs.text_embeds if self.feature_mode == "text" else outputs.multimodal_embeds
        )
        features = features.cpu().numpy()
        assert model.max_txt_len == 32
        assert features.shape[1] == model.max_txt_len, f"{features.shape}"

        img_ids = samples["image_id"].cpu().numpy()
        for feature, img_id in zip(features, img_ids):
            # for h5py, key should be 'str'
            results.append({"image_id": str(img_id), "feature": feature})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=self.target_dir,
            filename="{}_feature_{}".format(self.feature_mode, split_name),
            remove_duplicate="image_id",
        )
        return {"agg_metrics": 0.0}

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        print("Saving Qformer features...")
        # we assume this task is called with distributed w/ 1 gpu
        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            result_dict = dict_by_image_id(result)
            final_result_file = os.path.join(result_dir, "%s.h5" % filename)

            h = h5py.File(final_result_file, "w")
            for k, v in result_dict.items():
                h.create_dataset(k, data=v)
            h.close()
            print(f"result file w/ {len(result)} elements saved to {final_result_file}...")
        return final_result_file

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        data_root = run_cfg.data_root
        target_dir = os.path.join(data_root, "shutterstock", "caption_features")
        os.makedirs(target_dir, exist_ok=True)

        feature_mode = run_cfg.feature_mode
        assert feature_mode in ["text", "multimodal"]
        return cls(
            feature_mode=feature_mode,
            target_dir=target_dir,
        )


def dict_by_image_id(feats_list):
    feats_dict = {}
    for ann in feats_list:
        feats_dict[ann["image_id"]] = ann["feature"]
    return feats_dict
