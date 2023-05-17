from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import load_dataset_config
from nice.datasets.builders.nice_builder import NICEBuilder
from nice.datasets.builders.shutterstock_builder import ShutterStockBuilder

__all__ = ["NICEBuilder", "ShutterStockBuilder"]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(f"Dataset {name} not found. ")
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
            data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset
