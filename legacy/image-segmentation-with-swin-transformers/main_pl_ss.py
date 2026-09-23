import torch
import torchvision
import os
import numpy as np
from models.build_model import build_model
import torch.nn as nn
from datetime import datetime
import yaml
from dataset.dataset_semantic_segmentation import Dataset_Semantic_Segmentation
import pytorch_lightning as pl
from aux_train.LitAutoEncoder_SS import LitAutoEncoder
from pytorch_lightning import loggers as pl_loggers
import warnings
from operator import xor
from pathlib import Path
from dataset.dataset_potsdam_semantic_segmentation import DatasetPotsdamSemantiSegmentatin
from torch.utils.data import Dataset, DataLoader

def main():
    warnings.filterwarnings('ignore')
    with open('config/train_pl_ss.yaml') as file:
        # scalar values to Python the dictionary format
        config = yaml.safe_load(file)
        #print(config)

    BATCH_SIZE      = config["batch_size"]
    epochs          = config["epochs"]
    model_name      = config["model"]
    vq_flag         = config["vq_flag"]
    train_data_path = config["train_data_path"]
    valid_data_path = config["valid_data_path"]
    learning_rate   = config["learning_rate"]
    models          = config["model_name"]
    ckpt_path       = config["ckpt_path"]
    logdir_path     = config["logdir_path"]
    pretrain        = config["pretrain"]
    dataset_name    = config["dataset_name"]

    train_dataset_length  = config["train_dataset_length"]
    valid_dataset_length  = config["valid_dataset_length"]


    assert not xor("vq" in model_name,  vq_flag), "[PARAMETERS ERROR] \t Mismatch model and vq_flag, model name : {}, vq_flag : {} ".format(model_name,vq_flag)
    assert model_name in models, "[PARAMETERS ERROR] \t Model name {} not in available model lists \n\n[{}]\n\n".format(model_name, models)
    assert dataset_name == "Vaihingen" or dataset_name == "Potsdam", "[ERROR] Wrong dataset selected..."

    # <------------------------------------------------------------------------>
    #Create and configure logger
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    # <------------------------------------------------------------------------>
    if torch.cuda.is_available():
        print()
        print("@"*50)
        print("\n",flush=True)

        os.system('nvidia-smi')

    print()
    print("@"*50)
    print("\n",flush=True)

    print("Loading reference batch of 8 samples...")

    if dataset_name == "Vaihingen":
        valid_dataset = Dataset_Semantic_Segmentation(  data_path=valid_data_path,
                                                        size_w=224,
                                                        size_h=224,
                                                        batch_size=BATCH_SIZE,
                                                        transform = None
                                                        )

        valid_dataloader = valid_dataset.data_iter_index(8)
    elif dataset_name == "Potsdam":
        valid_dataset = DatasetPotsdamSemantiSegmentatin( data_path=valid_data_path,
                                                size_w=224,
                                                size_h=224,
                                                transform = None,
                                                dataset_length=8
                                                )
        valid_dataloader =  DataLoader(dataset=valid_dataset, batch_size=8)

    reference_batch = next(iter(valid_dataloader))
    ref_im, ref_gt = reference_batch

    print("Reference batch of 8 samples loaded\n\n",flush=True)

    print()
    print("@"*50)
    print("\n",flush=True)


    # <------------------------------------------------------------------------>
    # BUILD MODELS
    # <------------------------------------------------------------------------>



    # model
    model = LitAutoEncoder( model_name=model_name,
                            vq_flag = vq_flag,
                            ref_im = ref_im, ref_gt = ref_gt,
                            train_dataset_length=train_dataset_length,valid_dataset_length=valid_dataset_length,
                            train_data_path=train_data_path, valid_data_path=valid_data_path,
                            learning_rate = learning_rate,
                            batch_size = BATCH_SIZE,
                            dataset_name = dataset_name)

    os.makedirs('pytorch_ligthening_logs/logs', exist_ok=True)
    drive_logs_folder =  "pytorch_ligthening_logs/logs"

    if not logdir_path == "None":
        drive_logs_folder = os.path.join(logdir_path,drive_logs_folder)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=drive_logs_folder, name="{}_logs".format(model_name))

    # training

    if torch.cuda.is_available():

        if pretrain:
            trainer = pl.Trainer(
                                    max_epochs=epochs,
                                    gpus=1,
                                    logger = tb_logger,
                                    auto_scale_batch_size="binsearch", #   None | "power" | "binsearch"
                                    default_root_dir=drive_logs_folder,
                                    resume_from_checkpoint=ckpt_path
                                )
        else:
            trainer = pl.Trainer(
                                    max_epochs=epochs,
                                    gpus=1,
                                    logger = tb_logger,
                                    auto_scale_batch_size="binsearch", #   None | "power" | "binsearch"
                                    default_root_dir=drive_logs_folder,
                                )
    else:
        if pretrain:
            trainer = pl.Trainer(
                                    max_epochs=epochs,
                                    logger = tb_logger,
                                    auto_scale_batch_size="binsearch", #   None | "power" | "binsearch"
                                    default_root_dir=drive_logs_folder,
                                    resume_from_checkpoint=ckpt_path
                                )
        else:
            trainer = pl.Trainer(
                                    max_epochs=epochs,
                                    logger = tb_logger,
                                    auto_scale_batch_size="binsearch", #   None | "power" | "binsearch"
                                    default_root_dir=drive_logs_folder,
                                )

    print()
    print("@"*50)
    print("\n",flush=True)

    res = trainer.tune(model)
    model.hparams.batch_size = res["scale_batch_size"]

    fle = Path('batch_size_log.txt')
    fle.touch(exist_ok=True)
    with open("batch_size_log.txt","a") as f:
        f.write("model : {}, batch_size selected : {}\n".format(model_name,model.hparams.batch_size))
        f.close()

    print("\n\nBatch size set to : {}\n\n".format(model.hparams.batch_size), flush=True)

    print()
    print("@"*50)
    print("\n",flush=True)

    trainer.fit(model)

    print()
    print("@"*50)
    print("\n",flush=True)

    print("TRAIN COMPLETED")

    print()
    print("@"*50)
    print("\n",flush=True)

    # <------------------------------------------------------------------------>
    #   END TRAINING FUNCTION
    # <------------------------------------------------------------------------>


if __name__ == "__main__":
    main()
