import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
from datetime import datetime
import yaml
import pytorch_lightning as pl
import warnings
from operator import xor

from models.build_model import build_model
from aux_train.LitFeaturesExtractor import LitFeaturesExtractor
from pytorch_lightning import loggers as pl_loggers
from aux_train.LitAutoEncoder_SS import LitAutoEncoder
from dataset.dataset_semantic_segmentation import Dataset_Semantic_Segmentation
from pathlib import Path
from dataset.dataset_potsdam_semantic_segmentation import DatasetPotsdamSemantiSegmentatin
from torch.utils.data import Dataset, DataLoader
import copy
import gc


def main():

    warnings.filterwarnings('ignore')

    with open('config/train_pl_features_extractor.yaml') as file:
        # scalar values to Python the dictionary format
        config = yaml.safe_load(file)

    # <------------------------------------------------------------------------>

    model_name_features_extractor       = config["model_features_extractor"]
    models_features_extractor           = config["models_features_extractor"]

    model_name_semantic_segmentation    = config["model_semantic_segmentation"]
    models_semantic_segmentation        = config["models_semantic_segmentation"]

    BATCH_SIZE      = config["batch_size"]
    epochs          = config["epochs"]
    vq_flag         = config["vq_flag"]
    train_data_path = config["train_data_path"]
    valid_data_path = config["valid_data_path"]
    learning_rate   = config["learning_rate"]
    ckpt_path       = config["ckpt_path"]
    logdir_path     = config["logdir_path"]
    pretrain        = config["pretrain"]
    dataset_name    = config["dataset_name"]

    train_dataset_length  = config["train_dataset_length"]
    valid_dataset_length  = config["valid_dataset_length"]



    # <------------------------------------------------------------------------>

    #check on features extractor
    assert model_name_features_extractor in models_features_extractor, "[PARAMETERS ERROR] \t Model {} does not exists".format(model_name_features_extractor)
    assert not xor("vq" in model_name_features_extractor,  vq_flag), "[PARAMETERS ERROR] \t Mismatch model and vq_flag, model name : {}, vq_flag : {} ".format(model_name_features_extractor,vq_flag)
    assert not "-ss" in model_name_features_extractor, "[PARAMETERS ERROR] \t Wrong model selected"

    #check on semantic segmentation
    assert model_name_semantic_segmentation in models_semantic_segmentation, "[PARAMETERS ERROR] \t Model {} does not exists".format(model_name_semantic_segmentation)
    assert not xor("vq" in model_name_semantic_segmentation,  vq_flag), "[PARAMETERS ERROR] \t Mismatch model and vq_flag, model name : {}, vq_flag : {} ".format(model_name_semantic_segmentation,vq_flag)
    assert "-ss" in model_name_semantic_segmentation, "[PARAMETERS ERROR] \t Wrong model selected"

    assert dataset_name == "Vaihingen" or dataset_name == "Potsdam", "[ERROR] Wrong dataset selected..."
    # <------------------------------------------------------------------------>
    #Create and configure logger
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    # <------------------------------------------------------------------------>
    if torch.cuda.is_available():
        print()
        print("@"*60)
        print("\n",flush=True)
        os.system('nvidia-smi')

    # <------------------------------------------------------------------------>
    # BUILD MODELS FOR DINO
    # <------------------------------------------------------------------------>

    print()
    print("@"*60)
    print("\n",flush=True)

    # model
    model_features_extractor = LitFeaturesExtractor(
                                    model_name=model_name_features_extractor,
                                    vq_flag=vq_flag,
                                    train_data_path=train_data_path,
                                    valid_data_path=valid_data_path,
                                    learning_rate = learning_rate,
                                    batch_size = BATCH_SIZE,
                                    train_dataset_length=train_dataset_length,
                                    valid_dataset_length=valid_dataset_length,
                                    local_crops_number = 6,
                                    nepochs=epochs,
                                    dataset_name = dataset_name
                                 )


    os.makedirs('pytorch_ligthening_logs/logs', exist_ok=True)
    drive_logs_folder =  "pytorch_ligthening_logs/logs"

    if not logdir_path == "None":
        drive_logs_folder = os.path.join(logdir_path,drive_logs_folder)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=drive_logs_folder, name="{}_FE_dino_logs".format(model_name_features_extractor))

    print()
    print("@"*60)
    print("\n",flush=True)

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
    print("@"*60)
    print("\n",flush=True)

    res = trainer.tune(model_features_extractor)

    print()
    print("@"*60)
    print("\n",flush=True)

    model_features_extractor.hparams.batch_size = res["scale_batch_size"]

    print("\n\nBatch size set to : {}".format(model_features_extractor.hparams.batch_size), flush=True)
    print("Alpha Cosine Schedule set to : {}\n\n".format(model_features_extractor._alpha_cosine_schedule), flush=True)

    fle = Path('batch_size_log.txt')
    fle.touch(exist_ok=True)
    with open("batch_size_log.txt","a") as f:
        f.write("model DINO : {}, batch_size selected : {}\n".format(model_name_features_extractor,model_features_extractor.hparams.batch_size))
        f.close()

    print()
    print("@"*60)
    print("\n",flush=True)

    trainer.fit(model_features_extractor)

    print()
    print("@"*60)
    print("\n",flush=True)

    #print("TRAIN COMPLETED")

    # <------------------------------------------------------------------------>
    #   END TRAINING FEATURES EXTRACTOR
    # <------------------------------------------------------------------------>

    print()
    print("@"*60)
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
    print("@"*60)
    print("\n",flush=True)

    print("<------------------------------------------------------------------------>")
    print("\t BUILD MODELS FOR TRANSFER LEARNING")
    print("<------------------------------------------------------------------------>")

    print()
    print("@"*60)
    print("\n",flush=True)

    # model
    model_semantic_segmentation = LitAutoEncoder(
                                        model_name=model_name_semantic_segmentation,
                                        vq_flag = vq_flag,
                                        ref_im = ref_im, ref_gt = ref_gt,
                                        train_dataset_length=train_dataset_length,valid_dataset_length=valid_dataset_length,
                                        train_data_path=train_data_path, valid_data_path=valid_data_path,
                                        learning_rate = learning_rate,
                                        batch_size = BATCH_SIZE,
                                        dataset_name = dataset_name
                                        )

    # changing last layer of the fetures extractor to adapts its to LitAutoEncoder
    model_features_extractor.student_model.finalconv3 =  copy.deepcopy(model_semantic_segmentation.model.finalconv3)

    # Replace the model in LitAutoEncoder with a copy of the already trained one

    model_semantic_segmentation.model = copy.deepcopy(model_features_extractor.student_model)
    model_semantic_segmentation.model.load_state_dict(model_features_extractor.student_model.state_dict())

    # delete features extractor model to free GPU space
    del model_features_extractor
    gc.collect()
    torch.cuda.empty_cache()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=drive_logs_folder, name="{}_TL_dino_logs".format(model_name_semantic_segmentation))

    print()
    print("@"*60)
    print("\n",flush=True)

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
    print("@"*60)
    print("\n",flush=True)

    res = trainer.tune(model_semantic_segmentation)

    print()
    print("@"*60)
    print("\n",flush=True)

    model_semantic_segmentation.hparams.batch_size = res["scale_batch_size"]

    print("\n\nBatch size set to : {}".format(model_semantic_segmentation.hparams.batch_size), flush=True)

    fle = Path('batch_size_log.txt')
    fle.touch(exist_ok=True)
    with open("batch_size_log.txt","a") as f:
        f.write("model DINO : {}, batch_size selected : {}\n\n".format(model_name_semantic_segmentation,model_semantic_segmentation.hparams.batch_size))
        f.close()

    print()
    print("@"*60)
    print("\n",flush=True)

    trainer.fit(model_semantic_segmentation)

    print()
    print("@"*60)
    print("\n",flush=True)

    #print("TRAIN COMPLETED")

    print("<------------------------------------------------------------------------>")
    print("\t END TRAINING SEMANTIC SEGMENTATION MODEL WITH DINO PRE-TRAINING")
    print("<------------------------------------------------------------------------>")

    print()
    print("@"*60)
    print("\n",flush=True)


if __name__ == "__main__":
    main()
