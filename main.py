# %%
import json
import multiprocessing
import os
import threading
import time

import numpy as np
import tensorflow as tf
import torch

from config import config
from data_loading.vae_pu_dataloaders import create_vae_pu_adapter, get_dataset
from external.LBE import eval_LBE, train_LBE
from external.nnPUlearning.api import nnPU
from external.nnPUss.api import nnPUv2
from external.sar_experiment import SAREMThreadedExperiment
from external.two_step import eval_2_step, train_2_step
from vae_pu_occ.vae_pu_occ_trainer import VaePuOccTrainer

label_frequencies = [0.9, 0.7, 0.5, 0.3, 0.1, 0.02]
# label_frequencies = [0.5, 0.02]
# label_frequencies = [0.7, 0.5, 0.3, 0.02]
# label_frequencies = [0.3, 0.7]
# label_frequencies = [0.02, 0.5]

start_idx = 0
num_experiments = 10
epoch_multiplier = 1

datasets = [
    "MNIST 3v5",
    "CIFAR CarTruck",
    "STL MachineAnimal",
    "CDC-Diabetes",
    "MNIST OvE",
    "CIFAR MachineAnimal",
]

# config["data"] = "MNIST 3v5"
# config['data'] = 'CIFAR CarTruck'
# config['data'] = 'STL MachineAnimal'
# config['data'] = 'Gas Concentrations'
# config["data"] = "MNIST OvE"
# config['data'] = 'CIFAR MachineAnimal'

# config["data"] = "MNIST 3v5 SCAR"
# config["data"] = "MNIST OvE SCAR"
# config['data'] = 'STL MachineAnimal SCAR'

if "SCAR" in config["data"]:
    config["use_SCAR"] = True
else:
    config["use_SCAR"] = False

# config["occ_methods"] = ["OC-SVM", "IsolationForest", "ECODv2", "A^3"]
# config["occ_methods"] = ["MixupPU", "EM-PU", "MixupPU+concat"]
# config["occ_methods"] = ["MixupPU+concat"]
config["occ_methods"] = [
    # "OC-SVM",
    "IsolationForest",
    # "ECODv2",
    "A^3",
    # "OddsRatio-e100-lr1e-3",
    # "OddsRatio-e200-lr1e-4",
    # "OddsRatio-PUprop-e100-lr1e-3",
    "OddsRatio-PUprop-e200-lr1e-4",
    # "OddsRatio-e100-lr1e-3-ES",
    # "OddsRatio-e200-lr1e-3-ES",
    # "OddsRatio-e50-lr1e-3",
    # "OddsRatio-e200-lr1e-3",
    # "OddsRatio-e100-lr1e-2",
    # "OddsRatio-e50-lr1e-2",
    # "OddsRatio-e200-lr1e-2",
    # "OddsRatio-e100-lr1e-4",
    # "OddsRatio-e50-lr1e-4",
    # "OddsRatio-e200-lr1e-4-ES",
    # "OddsRatio-e200-lr1e-5-ES",
    # "nnPU",
    # "nnPUcc",
    # "VAE-PU+nnPU",
    # "MixupPU",
    # "MixupPU-NJW",
    # "MixupPU-NJW-no-norm",
    # "MixupPU-no-name",
    # "MixupPU-no-name-no-norm",
    # "MixupPU+extra-loss-3",
    # "MixupPU+extra-loss-1",
    # "MixupPU+extra-loss-0.3",
    # "MixupPU+extra-loss-0.1",
    # "MixupPU+extra-loss-0.03",
    # "MixupPU+extra-loss-0.003",
    # "MixupPU+extra-loss-log-3",
    # "MixupPU+extra-loss-log-1",
    # "MixupPU+extra-loss-log-0.3",
    # "MixupPU+extra-loss-log-0.1",
    # "MixupPU+extra-loss-log-0.03",
    # "MixupPU+extra-loss-log-0.003",
]
# config["occ_methods"] = ["EM-PU"]

config["use_original_paper_code"] = False
# config['use_original_paper_code'] = True
config["use_old_models"] = True
# config["use_old_models"] = False

config["training_mode"] = "VAE-PU"
# config['training_mode'] = 'SAR-EM'
# config['training_mode'] = 'LBE'
# config["training_mode"] = "2-step"
# config["training_mode"] = "nnPU"
# config["training_mode"] = "nnPUss"
# config["training_mode"] = "uPU"

config["vae_pu_variant"] = None

config["nnPU_beta"], config["nnPU_gamma"] = None, None
# config["nnPU_beta"], config["nnPU_gamma"] = 0, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-3, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-2, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-4, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-3, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-2, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-4, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-3, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-2, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-4, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-3, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-2, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-4, 0.5

if config["nnPU_beta"] is not None and config["nnPU_gamma"] is not None:
    config["vae_pu_variant"] = (
        f"beta_{config['nnPU_beta']:.0e}_gamma_{config['nnPU_gamma']:.0e}"
    )


config["train_occ"] = True
config["occ_num_epoch"] = round(100 * epoch_multiplier)

config["early_stopping"] = True
config["early_stopping_epochs"] = 10

if config["use_original_paper_code"]:
    config["mode"] = "near_o"
else:
    config["mode"] = "near_y"

config["device"] = "auto"

# used by SAR-EM
n_threads = multiprocessing.cpu_count()
sem = threading.Semaphore(n_threads)
threads = []

for dataset in datasets:
    config["data"] = dataset

    for training_mode in [
        "VAE-PU",
        # "LBE",
        # "SAR-EM",
        # "nnPU",
        # "nnPUv2",
        # "nnPUss",
        # "nnPUssv2",
    ]:
        config["training_mode"] = training_mode
        for idx in range(start_idx, start_idx + num_experiments):
            for base_label_frequency in label_frequencies:
                config["base_label_frequency"] = base_label_frequency

                np.random.seed(idx)
                torch.manual_seed(idx)
                tf.random.set_seed(idx)

                if config["device"] == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                (
                    train_samples,
                    val_samples,
                    test_samples,
                    label_frequency,
                    pi_p,
                    n_input,
                ) = get_dataset(
                    config["data"],
                    device,
                    base_label_frequency,
                    use_scar_labeling=config["use_SCAR"],
                )
                vae_pu_data = create_vae_pu_adapter(
                    train_samples, val_samples, test_samples, device
                )

                config["label_frequency"] = label_frequency
                config["pi_p"] = pi_p
                config["n_input"] = n_input

                config["pi_pl"] = label_frequency * pi_p
                config["pi_pu"] = pi_p - config["pi_pl"]
                config["pi_u"] = 1 - config["pi_pl"]

                batch_size = 1000
                pl_batch_size = int(np.ceil(config["pi_pl"] * batch_size))
                u_batch_size = batch_size - pl_batch_size
                config["batch_size_l"], config["batch_size_u"] = (
                    pl_batch_size,
                    u_batch_size,
                )
                config["batch_size_l_pn"], config["batch_size_u_pn"] = (
                    pl_batch_size,
                    u_batch_size,
                )

                config["n_h_y"] = 10
                config["n_h_o"] = 2
                config["lr_pu"] = 3e-4
                config["lr_pn"] = 1e-5

                if config["data"] == "CDC-Diabetes":
                    epoch_multiplier = 0.5

                config["num_epoch_pre"] = round(100 * epoch_multiplier)
                config["num_epoch_step1"] = round(400 * epoch_multiplier)
                config["num_epoch_step_pn1"] = round(500 * epoch_multiplier)
                config["num_epoch_step_pn2"] = round(600 * epoch_multiplier)
                config["num_epoch_step2"] = round(500 * epoch_multiplier)
                config["num_epoch_step3"] = round(700 * epoch_multiplier)
                config["num_epoch"] = round(800 * epoch_multiplier)

                config["n_hidden_cl"] = []
                config["n_hidden_pn"] = [300, 300, 300, 300]

                if config["data"] == "MNIST OvE":
                    config["alpha_gen"] = 0.1
                    config["alpha_disc"] = 0.1
                    config["alpha_gen2"] = 3
                    config["alpha_disc2"] = 3
                elif config["data"] == "CDC-Diabetes":
                    config["alpha_gen"] = 0.01
                    config["alpha_disc"] = 0.01
                    config["alpha_gen2"] = 0.3
                    config["alpha_disc2"] = 0.3

                    config["n_h_y"] = 5

                    config["n_hidden_pn"] = [200, 200]
                    config["n_hidden_vae_e"] = [100, 100]
                    config["n_hidden_vae_d"] = [100, 100]
                    config["n_hidden_disc"] = [20]

                    config["lr_pu"] = 1e-5
                    config["lr_pn"] = 3e-5
                elif ("CIFAR" in config["data"] or "STL" in config["data"]) and config[
                    "use_SCAR"
                ]:
                    config["alpha_gen"] = 3
                    config["alpha_disc"] = 3
                    config["alpha_gen2"] = 1
                    config["alpha_disc2"] = 1
                    ### What is it?
                    config["alpha_test"] = 1.0
                elif (
                    "CIFAR" in config["data"] or "STL" in config["data"]
                ) and not config["use_SCAR"]:
                    config["alpha_gen"] = 0.3
                    config["alpha_disc"] = 0.3
                    config["alpha_gen2"] = 1
                    config["alpha_disc2"] = 1
                    ### What is it?
                    config["alpha_test"] = 1.0
                else:
                    config["alpha_gen"] = 1
                    config["alpha_disc"] = 1
                    config["alpha_gen2"] = 10
                    config["alpha_disc2"] = 10

                config["device"] = device
                config["directory"] = os.path.join(
                    "result",
                    config["data"],
                    str(base_label_frequency),
                    "Exp" + str(idx),
                )

                if config["training_mode"] == "VAE-PU":
                    trainer = VaePuOccTrainer(
                        num_exp=idx, model_config=config, pretrain=True
                    )
                    trainer.train(vae_pu_data)
                else:
                    np.random.seed(idx)
                    torch.manual_seed(idx)
                    tf.random.set_seed(idx)
                    method_dir = os.path.join(
                        config["directory"], "external", config["training_mode"]
                    )

                    if config["training_mode"] == "SAR-EM":
                        exp_thread = SAREMThreadedExperiment(
                            train_samples,
                            test_samples,
                            idx,
                            base_label_frequency,
                            config,
                            method_dir,
                            sem,
                        )
                        exp_thread.start()
                        threads.append(exp_thread)
                    else:
                        if config["training_mode"] == "LBE":
                            log_prefix = f"Exp {idx}, c: {base_label_frequency} || "

                            lbe_training_start = time.perf_counter()
                            lbe = train_LBE(
                                train_samples,
                                val_samples,
                                verbose=True,
                                log_prefix=log_prefix,
                            )
                            lbe_training_time = time.perf_counter() - lbe_training_start

                            accuracy, precision, recall, f1 = eval_LBE(
                                lbe, test_samples, verbose=True, log_prefix=log_prefix
                            )
                        elif config["training_mode"] == "2-step":
                            training_start = time.perf_counter()
                            clf = train_2_step(train_samples, config, idx)
                            training_time = time.perf_counter() - training_start

                            accuracy, precision, recall, f1 = eval_2_step(
                                clf, test_samples
                            )
                        elif config["training_mode"] in ["nnPUss", "nnPU", "uPU"]:
                            x, y, s = train_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            train_samples = x, y, s

                            training_start = time.perf_counter()
                            clf = nnPU(model_name=config["training_mode"])
                            clf.train(train_samples, config["pi_p"])
                            training_time = time.perf_counter() - training_start

                            x, y, s = test_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            test_samples = x, y, s

                            accuracy, precision, recall, f1 = clf.evaluate(test_samples)
                        elif config["training_mode"] in ["nnPUssv2", "nnPUv2"]:
                            x, y, s = train_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            train_samples = x, y, s

                            training_start = time.perf_counter()
                            clf = nnPUv2(model_name=config["training_mode"])
                            clf.train(
                                train_samples,
                                val_samples=test_samples,
                                pi=config["pi_p"],
                            )
                            training_time = time.perf_counter() - training_start

                            x, y, s = test_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            test_samples = x, y, s

                            accuracy, precision, recall, f1 = clf.evaluate(test_samples)

                        metric_values = {
                            "Method": config["training_mode"],
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1 score": f1,
                            "Time": training_time,
                        }

                        os.makedirs(method_dir, exist_ok=True)
                        with open(
                            os.path.join(method_dir, "metric_values.json"), "w"
                        ) as f:
                            json.dump(metric_values, f)

        for t in threads:
            t.join()

# %%
