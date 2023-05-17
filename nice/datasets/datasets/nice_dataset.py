import os

import numpy as np
import torch
from PIL import Image, ImageFile

import h5py
from lavis.datasets.datasets.caption_datasets import BaseDataset, CaptionEvalDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _get_str_caption(caption):
    if isinstance(caption, list):
        caption = caption[0]
    assert isinstance(caption, str)
    return caption


def _get_img_id_from_name(image_name):
    return int(image_name.split("/")[-1].split(".jpg")[0])


def _load_ret_features(ret_feature_path):
    print("loading value features of retrieved samples from: {}..".format(ret_feature_path))
    return h5py.File(ret_feature_path, 'r')


def _get_ret_features(ret_features, ret_image_ids, max_num):
    # ret_image_ids: retrived image ids of shutterstock for each query (nice) sample
    features = []
    for ret_img_id in ret_image_ids[:max_num]:  # maximum 50 in annotations
        features.append(ret_features[str(ret_img_id)][:])
    assert len(features) == max_num
    return torch.from_numpy(np.stack(features, axis=0))  # (M, 32, 768)


class NICETrainDataset(BaseDataset):

    def __init__(
        self, vis_processor, text_processor, vis_root, ann_paths, ret_feature=None, max_ret=None
    ):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.ret_features = None
        if ret_feature is not None:
            self.ret_features = _load_ret_features(ret_feature)
        self.max_ret = max_ret if max_ret is not None else 10  # maximum t0

    def close_ret_features(self):
        if self.ret_features is not None:
            self.ret_features.close()

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_id = _get_img_id_from_name(ann["image"])
        image_path = os.path.join(self.vis_root, ann["image"])

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        caption = self.text_processor(_get_str_caption(ann["caption"]))

        data_dict = {"image": image, "text_input": caption, "image_id": img_id}
        if self.ret_features is not None:
            # (M, L, D); M: number of retrieved samples
            data_dict["ret_features"] = _get_ret_features(
                self.ret_features, ann["ret_image_ids"], max_num=self.max_ret
            )
        return data_dict


class NICEEvalDataset(CaptionEvalDataset):

    def __init__(
        self, vis_processor, text_processor, vis_root, ann_paths, ret_feature=None, max_ret=None
    ):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.ret_features = None
        if ret_feature is not None:
            self.ret_features = _load_ret_features(ret_feature)
        self.max_ret = max_ret if max_ret is not None else 10  # maximum t0

    def close_ret_features(self):
        if self.ret_features is not None:
            self.ret_features.close()

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_id = img_id = _get_img_id_from_name(ann["image"])
        image_path = os.path.join(self.vis_root, ann["image"])

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        data_dict = {"image": image, "image_id": img_id, "instance_id": ann["instance_id"]}
        if "caption" in ann:
            caption = self.text_processor(_get_str_caption(ann["caption"]))
            data_dict["text_input"] = caption

        if self.ret_features is not None:
            data_dict["ret_features"] = _get_ret_features(
                self.ret_features, ann["ret_image_ids"], max_num=self.max_ret
            )

        return data_dict
