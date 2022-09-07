from copy import deepcopy

#### ASAP ####


def get_asap_normal(update: dict):
    configs = deepcopy(asap_normal)
    for _, config in configs.items():
        config.update(update)
    return configs


asap_common_settings = {
    "task;debug_mode": False,
    "data;dataset": "asap",
    "data;use_validation_set": True,
    "data;acc_val": True,
    "data;num_batches": 10,
    "arch;model": "aes_cwm",
    "train;num_epochs": 10,
    "log;note": "QUERY v in ['', 'none']",
}
asap_normal = {
    "ft": {
        **asap_common_settings,
        "train;trainer": "fine_tune",
        "train;num_fine_tune_epochs": 0,
        "task;total_num_exemplars": 0,
    },
    "ft_reg": {
        **asap_common_settings,
        "train;trainer": "fine_tune_reg",
        "train;num_fine_tune_epochs": 0,
        "task;total_num_exemplars": 0,
    },
    "lwf": {
        **asap_common_settings,
        "train;trainer": "lwf",
        "train;num_fine_tune_epochs": 0,
        "task;total_num_exemplars": 0,
    },
    "icarl": {
        **asap_common_settings,
        "train;trainer": "icarl",
        "task;total_num_exemplars": 100,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
    },
    "eeil": {
        **asap_common_settings,
        "train;trainer": "eeil",
        "task;total_num_exemplars": 100,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 5,
    },
    "lucir": {
        **asap_common_settings,
        "train;trainer": "lucir",
        "task;total_num_exemplars": 100,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
    },
    "podnet": {
        **asap_common_settings,
        "train;trainer": "podnet",
        "task;total_num_exemplars": 100,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 5,
    },
    "lucir_cwd": {
        **asap_common_settings,
        "train;trainer": "lucir_cwd",
        "task;total_num_exemplars": 100,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
        "train;loss_args;fixed_lambda": 0.1,
    },
    "eeol": {
        **asap_common_settings,
        "task;smooth_factor": 6.0,
        "train;trainer": "eeol",
        "task;total_num_exemplars": 100,
        "task;cal_num_exemplars_per_class_method": "by_num",
        "train;num_fine_tune_epochs": 0,
        "train;loss_args;t": 0.3,
        "train;loss_args;suppress_factor_at_2t": 0.1,
        "train;loss_args;reject_order": "linear",
    },
}

asap_num_e = {
    "eeil": {
        **asap_common_settings,
        "train;trainer": "eeil",
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 5,
    },
    "eeol": {
        **asap_common_settings,
        "task;smooth_factor": 5.0,
        "train;trainer": "eeol",
        "task;cal_num_exemplars_per_class_method": "by_num",
        "train;num_fine_tune_epochs": 0,
        "train;loss_args;t": 0.1,
        "train;loss_args;suppress_factor_at_2t": 0.2,
    },
}

asap_runtime = {
    "joint": {
        **asap_common_settings,
        "train;trainer": "joint",
        "train;num_fine_tune_epochs": 0,
        "task;total_num_exemplars": 0,
    },
    "eeol": {
        **asap_common_settings,
        "task;smooth_factor": 6.0,
        "train;trainer": "eeol",
        "task;total_num_exemplars": 100,
        "task;cal_num_exemplars_per_class_method": "by_num",
        "train;num_fine_tune_epochs": 0,
        "train;loss_args;t": 0.3,
        "train;loss_args;suppress_factor_at_2t": 0.1,
        "train;loss_args;reject_order": "linear",
    },
}
#### Cifar-100 ####
cifar_common_settings = {
    "task;debug_mode": False,
    "data;dataset": "cifar",
    "data;use_validation_set": False,
    "data;num_batches": 20,
    "arch;model": "resnet32",
    "train;num_epochs": 200,
    "log;note": "",
    "train;plateau_metric": "train_acc",
    "train;lr_scheduler_args;mode": "max",
}

cifar_normal = {
    "ft": {
        **cifar_common_settings,
        "train;trainer": "fine_tune",
        "train;num_fine_tune_epochs": 0,
    },
    "lwf": {
        **cifar_common_settings,
        "train;trainer": "lwf",
        "train;num_fine_tune_epochs": 0,
    },
    "icarl": {
        **cifar_common_settings,
        "train;trainer": "icarl",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
    },
    "eeil": {
        **cifar_common_settings,
        "train;trainer": "eeil",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 40,
        "train;loss_args;fixed_lambda": 0.5,
        # "log;note": "small_dist0.5",
    },
    "lucir": {
        **cifar_common_settings,
        "train;trainer": "lucir",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
    },
    "podnet": {
        **cifar_common_settings,
        "train;trainer": "podnet",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 40,
    },
    "lucir_cwd": {
        **cifar_common_settings,
        "train;trainer": "lucir_cwd",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
        "train;loss_args;fixed_lambda": 0.2,
    },
    "eeol": {
        **cifar_common_settings,
        "train;trainer": "eeol",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
        "train;loss_args;fixed_lambda": 0.5,
        # "log;note": "small_dist0.5",
    },
}

cifar_normal_numes = {
    "eeil": {
        **cifar_common_settings,
        "train;trainer": "eeil",
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 40,
    },
    "eeol": {
        **cifar_common_settings,
        "train;trainer": "eeol",
        "task;cal_num_exemplars_per_class_method": "by_num",
        "train;num_fine_tune_epochs": 0,
    },
}

cifar_common_settings_il = {
    "task;debug_mode": False,
    "data;dataset": "cifar",
    "data;use_validation_set": False,
    "arch;model": "resnet32",
    "train;num_epochs": 200,
    "train;plateau_metric": "train_acc",
    "train;lr_scheduler_args;mode": "max",
}
cifar_il = {
    "ft": {
        **cifar_common_settings_il,
        "train;trainer": "fine_tune",
        "train;num_fine_tune_epochs": 0,
    },
    "lwf": {
        **cifar_common_settings_il,
        "train;trainer": "lwf",
        "train;num_fine_tune_epochs": 0,
    },
    "icarl": {
        **cifar_common_settings_il,
        "train;trainer": "icarl",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
        "log;note": "bceall",
    },
    "eeil": {
        **cifar_common_settings_il,
        "train;trainer": "eeil",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 40,
    },
    "lucir": {
        **cifar_common_settings_il,
        "train;trainer": "lucir",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
    },
    "podnet": {
        **cifar_common_settings_il,
        "train;trainer": "podnet",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 40,
    },
    "lucir_cwd": {
        **cifar_common_settings_il,
        "train;trainer": "lucir_cwd",
        "task;num_exemplars_per_class": 20,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
        "train;loss_args;fixed_lambda": 0.5,
    },
}

#### Imagenet ####
imagenet_common_settings = {
    "task;debug_mode": False,
    "task;task": "olrandom",
    "data;dataset": "imagenet",
    "data;use_validation_set": False,
    "data;num_batches": 20,
    "arch;model": "resnet18",
    "train;num_epochs": 200,
    "log;note": "",
    "data;batch_size": 256,
}

imagenet_normal = {
    "ft": {
        **imagenet_common_settings,
        "train;trainer": "fine_tune",
        "train;num_fine_tune_epochs": 0,
    },
    "lwf": {
        **imagenet_common_settings,
        "train;trainer": "lwf",
        "train;num_fine_tune_epochs": 0,
    },
    "icarl": {
        **imagenet_common_settings,
        "train;trainer": "icarl",
        "task;num_exemplars_per_class": 50,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
    },
    "eeil": {
        **imagenet_common_settings,
        "train;trainer": "eeil",
        "task;num_exemplars_per_class": 50,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 40,
    },
    "lucir": {
        **imagenet_common_settings,
        "train;trainer": "lucir",
        "task;num_exemplars_per_class": 50,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
    },
    "podnet": {
        **imagenet_common_settings,
        "train;trainer": "podnet",
        "task;num_exemplars_per_class": 50,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 40,
    },
    "lucir_cwd": {
        **imagenet_common_settings,
        "train;trainer": "lucir_cwd",
        "task;num_exemplars_per_class": 50,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
        "train;loss_args;fixed_lambda": 0.5,
    },
    "eeol": {
        **imagenet_common_settings,
        "train;trainer": "eeol",
        "task;num_exemplars_per_class": 50,
        "task;cal_num_exemplars_per_class_method": "by_class",
        "train;num_fine_tune_epochs": 0,
        # "log;note": "small_dist0.5",
        "train;loss_args;fixed_lambda": 0.5,
    },
}
