from lavis.common.config import Config as _Config
from lavis.common.registry import registry
from omegaconf import OmegaConf

if "configuration" in registry.mapping["state"]:
    registry.mapping["state"].pop("configuration", None)


class Config(_Config):

    def __init__(self, args):
        self.config = {}

        self.args = args

        # Register the config and configuration for setup
        registry.register("configuration", self)

        user_config = self._build_opt_list(self.args.options)

        config = OmegaConf.load(self.args.cfg_path)

        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(
            config, data_type=user_config.pop("data_type", None)
        )

        self.config = OmegaConf.merge(runner_config, model_config, dataset_config, user_config)

    @staticmethod
    def build_dataset_config(config, data_type=None):
        datasets = config.get("datasets", None)
        if datasets is None:
            raise KeyError("Expecting 'datasets' as the root key for dataset configuration.")

        dataset_config = OmegaConf.create()
        if data_type is not None:
            assert len(datasets) == 1

        for dataset_name in datasets:
            builder_cls = registry.get_builder_class(dataset_name)

            if data_type is not None:
                datasets[dataset_name].type = data_type
            dataset_config_type = datasets[dataset_name].get("type", "default")
            dataset_config_path = builder_cls.default_config_path(type=dataset_config_type)

            # hiararchy override, customized config > default config
            dataset_config = OmegaConf.merge(
                dataset_config,
                OmegaConf.load(dataset_config_path),
                {"datasets": {
                    dataset_name: config["datasets"][dataset_name]
                }},
            )

        return dataset_config
