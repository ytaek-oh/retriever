import os
import warnings

import lavis.common.utils as utils
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from nice.datasets.datasets.nice_dataset import NICEEvalDataset, NICETrainDataset


@registry.register_builder("nice")
class NICEBuilder(BaseDatasetBuilder):
    train_dataset_cls = NICETrainDataset
    eval_dataset_cls = NICEEvalDataset

    DATASET_CONFIG_DICT = {
        # required for preprocessing steps
        "default": "configs/datasets/nice/defaults.yaml",
        "discovery_nice_test_top1": "configs/datasets/nice/discovery_nice_test_top1.yaml",

        # train with only 4k labeled nice dataset (validation 4k splits as training and remaining 1k for eval.)  # noqa
        "train_split_4k": "configs/datasets/nice/train_split_4k.yaml",
        "train_split_4k_with_fusion": "configs/datasets/nice/train_split_4k_with_fusion.yaml",

        # training with generated from dataset discovery (topk=1 per query)
        "train_discovery_top1": "configs/datasets/nice/train_discovery_top1.yaml",  # noqa
        "train_discovery_top1_with_fusion": "configs/datasets/nice/train_discovery_top1_with_fusion.yaml",  # noqa

        # training with generated from dataset discovery (topk=1 per query) and including valid 4k nice split  # noqa
        "train_discovery_top1_add_valid4k": "configs/datasets/nice/train_discovery_top1_add_valid4k.yaml",  # noqa
        "train_discovery_top1_add_valid4k_with_fusion": "configs/datasets/nice/train_discovery_top1_add_valid4k_with_fusion.yaml",  # noqa
    
        # final top scoring entry
        "train_discovery_top1_add_valid5k_with_fusion": "configs/datasets/nice/train_discovery_top1_add_valid5k_with_fusion.yaml",  # noqa
    }

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        max_ret = self.config.get("max_ret", None)
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

            # load retrieved feature info
            ret_feature_path = ann_info.get(split).get("ret_features", None)

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                ret_feature=ret_feature_path,
                max_ret=max_ret
            )

        return datasets
