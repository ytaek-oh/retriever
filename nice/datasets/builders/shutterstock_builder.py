import os
import warnings

import lavis.common.utils as utils
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from nice.datasets.datasets.shutterstock_image_text_pair import ShutterStock


@registry.register_builder("shutterstock")
class ShutterStockBuilder(BaseDatasetBuilder):
    eval_dataset_cls = ShutterStock

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/shutterstock/defaults.yaml",  # same as 1m
        "shutterstock_1m": "configs/datasets/shutterstock/shutterstock_1m.yaml",
        "extract_caption_features": "configs/datasets/shutterstock/extract_caption_features.yaml",
    }

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        print("Available dataset split keys: {}".format(", ".join(list(ann_info.keys()))))
        for split in ann_info.keys():
            is_train = split == "train"  # fixed

            # processors
            vis_processor = (
                self.vis_processors["train"] if is_train else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"] if is_train else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
            )

        return datasets
