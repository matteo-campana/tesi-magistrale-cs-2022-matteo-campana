import torch
import torchvision
from torchvision import transforms as T

import logging
import os
import numpy as np

from models.build_model import build_model
import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import yaml

from aux_train.train_class_semantic_segmentation import TrainModelSS
from aux_train.validation_class_semantic_segmentation import EvaluateModelSS
from dataset.dataset_semantic_segmentation import Dataset_Semantic_Segmentation
import sys

def main():

    '''
    BATCH_SIZE   = 4
    epochs          = 10
    dataset_length  = 4
    model_name      = "swin-transformer-vq-vae-ss"
    vq_flag         = False
    train_data_path = "/content/drive/MyDrive/Colab Notebooks/semantic_labelling/Dataset/Vaihingen_processed/train"
    valid_data_path = "/content/drive/MyDrive/Colab Notebooks/semantic_labelling/Dataset/Vaihingen_processed/validation"
    learning_rate   = 0.003
    '''

    with open('config/train_semantic_segmentation.yaml') as file:
        # scalar values to Python the dictionary format
        config = yaml.safe_load(file)
        #print(config)

    BATCH_SIZE      = config["batch_size"]
    epochs          = config["epochs"]
    dataset_length  = config["dataset_length"]
    model_name      = config["model"]
    vq_flag         = config["vq_flag"]
    train_data_path = config["train_data_path"]
    valid_data_path = config["valid_data_path"]
    learning_rate   = config["learning_rate"]

    pretrained_model    = config["pretrained_model_name"]
    models_weights_path = config["models_weights_path"]
    logdir_path         = config["logdir_path"]
    last_epoch          = config["last_epoch"]
    pretrain            = config["pretrain"]

    if "vq" in model_name:
        vq_flag = True
    else:
        vq_flag = False

    # <------------------------------------------------------------------------>
    #Create and configure logger
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    logging_file_name = "logging_file_{}.log".format(date)
    logging.basicConfig(filename="logging_file_name.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')

    #Creating an object
    logger=logging.getLogger()

    #Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)


    # <------------------------------------------------------------------------>

    logger.info("GPU availability : {}".format( torch.cuda.is_available() ) )
    if torch.cuda.is_available():
        os.system('nvidia-smi')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # <------------------------------------------------------------------------>
    # BUILD DATALOADERS
    # <------------------------------------------------------------------------>


    logger.info("initializing validation dataloader...")
    print("\ninitializing validation dataloader...")

    valid_dataset = Dataset_Semantic_Segmentation(  data_path=valid_data_path,
                                                    size_w=224,
                                                    size_h=224,
                                                    batch_size=BATCH_SIZE,
                                                    transform = None
                                                    )

    if dataset_length and (dataset_length > 0):
        valid_dataloader = valid_dataset.data_iter_index(dataset_length)
    else:
        valid_dataloader = valid_dataset.data_iter()

    logger.info("validation dataloader initialized")
    print("validation dataloader initialized\n")

    logger.info("initializing train dataloader...")
    print("\ninitializing train dataloader...")

    train_dataset = Dataset_Semantic_Segmentation(  data_path=train_data_path,
                                                    size_w=224,
                                                    size_h=224,
                                                    batch_size=BATCH_SIZE,
                                                    transform = None
                                                    )

    if dataset_length and (dataset_length > 0):
        train_dataloader = train_dataset.data_iter_index(dataset_length)
    else:
        train_dataloader = train_dataset.data_iter()

    logger.info("train dataloader initialized")
    print("train dataloader initialized\n")


    # <------------------------------------------------------------------------>
    # BUILD MODELS
    # <------------------------------------------------------------------------>

    logger.info("\ninitializing model...")
    print("initializing model...")

    model = build_model(model_name=model_name)

    last_epoch_model = last_epoch

    if pretrain and not pretrained_model == "None":
        model.load_state_dict(torch.load("{}/{}".format(models_weights_path,pretrained_model)))
        last_epoch_model = pretrained_model.replace("{}_weights_epoch_".format(model_name),"")
        last_epoch_model = last_epoch_model[:last_epoch_model.find("_")]
        last_epoch_model = int(last_epoch_model)


    logger.info("model initialized")
    print("model initialized\n")

    # <------------------------------------------------------------------------>

    logger.info("initializing train arguments...")
    print("\ninitializing train arguments...")

    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    print('Ignored')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    batch_size = BATCH_SIZE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("vector quantize flag set to : {}".format(vq_flag))

    logger.info("train arguments initialized")
    print("train arguments initialized\n")

    # <------------------------------------------------------------------------>
    # <------------------------------------------------------------------------>
    logger.info("moving model to device : {}...".format(device))
    print("\nmoving model to device : {}...".format(device))

    model = model.to(device)

    logger.info("model moved to device : {}".format(device))
    print("model moved to device : {}\n".format(device))

    # <------------------------------------------------------------------------>

    train_model = TrainModelSS( model = model,
                                vq_flag = vq_flag,
                                criterion = criterion,
                                optimizer = optimizer,
                                training = True)

    evaluate_model = EvaluateModelSS(   model = model,
                                        vq_flag = vq_flag,
                                        criterion = criterion,
                                        optimizer = optimizer,
                                        training = False)

    logger.info("train started...")
    print("\ntrain started...\n")

    if models_weights_path == "None":
        os.makedirs("models_weights_{}".format(model_name),exist_ok=True)
    else:
        os.makedirs(
            os.path.join(models_weights_path,
                         "models_weights_{}".format(model_name)
                         ),
                    exist_ok=True
                    )

    logger.info("tensorboard writer initialized...")
    print("\ntensorboard writer initialized...\n")

    # Writer will output to ./runs/ directory by default
    if logdir_path == "None":
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        writer = SummaryWriter("runs/run_{}_{}".format(model_name,date))
    else:
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        writer = SummaryWriter(os.path.join(logdir_path,"runs/run_{}_{}".format(model_name,date)))

    initial_epoch = 0
    if pretrain and last_epoch>= 0 and last_epoch == last_epoch_model:
        initial_epoch = last_epoch_model
    elif pretrain and last_epoch >= 1:
        initial_epoch = last_epoch
    elif pretrain:
        initial_epoch = last_epoch_model

    best_loss = sys.float_info.max

    for epoch in range(initial_epoch+1,epochs+1):

        # reset epoch metrics
        epoch_loss = 0
        mIoU_acc_epoch, pixel_acc_epoch, average_mse_epoch = 0, 0, 0
        epoch_recon_error, epoch_perplexity = 0, 0

        # <-------------------------------------------------------------------->
        # <-------------------------------------------------------------------->
        # TRAIN STEP
        for top_batch, gts_batch in tqdm(train_dataloader,desc = '[{}/{}] Training'.format(epoch,epochs)):

            #reset batch metrics
            batch_loss = 0
            mIoU_acc_batch , pixel_acc_batch , average_mse_batch = 0, 0, 0

            if vq_flag == True:
                batch_recon_error, batch_perplexity = 0, 0

            top_batch = top_batch.to(device)
            gts_batch = gts_batch.to(device)

            if vq_flag == True:
                batch_loss, batch_recon_error, batch_perplexity, mIoU_acc_batch , pixel_acc_batch, average_mse_batch  = train_model(top_batch, gts_batch)
                epoch_loss          += batch_loss
                epoch_recon_error   += batch_recon_error
                epoch_perplexity    += batch_perplexity

                mIoU_acc_epoch     += mIoU_acc_batch
                pixel_acc_epoch    += pixel_acc_batch
                average_mse_epoch  += average_mse_batch
            else:
                batch_loss, mIoU_acc_batch , pixel_acc_batch, average_mse_batch = train_model(top_batch, gts_batch)
                epoch_loss += batch_loss

                mIoU_acc_epoch     += mIoU_acc_batch
                pixel_acc_epoch    += pixel_acc_batch
                average_mse_epoch  += average_mse_batch
                average_mse_epoch  += average_mse_batch

        # <-------------------------------------------------------------------->
        if vq_flag == True:
            epoch_recon_error   /= len(train_dataloader)
            epoch_perplexity    /= len(train_dataloader)

            print("\t reconstruction error : {:4f}".format(epoch_recon_error))
            #print("\t epoch perplexity : {}".format(epoch_perplexity))

            logger.info("\t [epoch {}/{}] epoch reconstruction error : {}".format(epoch,epochs,epoch_recon_error))
            logger.info("\t [epoch {}/{}] epoch perplexity : {}".format(epoch,epochs,epoch_perplexity))

            # write on Tensorboard
            #writer.add_scalar('Perplexity/train', epoch_perplexity, epoch)
            writer.add_scalar('ReconstructionError/train', epoch_recon_error, epoch)

        epoch_loss          /= len(train_dataloader)
        mIoU_acc_epoch      /= len(train_dataloader)
        pixel_acc_epoch     /= len(train_dataloader)
        average_mse_epoch   /= len(train_dataloader)

        print("\t mIoU_acc_epoch     :  {:4f}".format(mIoU_acc_epoch))
        print("\t pixel_acc_epoch    :  {:4f}".format(pixel_acc_epoch))
        print("\t average_mse_epoch  :  {:4f}".format(average_mse_epoch))
        print("\t epoch loss         :  {:4f}".format(epoch_loss))

        print()
        logger.info("\t [epoch {}/{}] epoch loss : {}".format(epoch,epochs,epoch_loss))

        # write on Tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        writer.add_scalar('mIoU/train'           , mIoU_acc_epoch       , epoch)
        writer.add_scalar('pixel_acc/train'      , pixel_acc_epoch      , epoch)
        writer.add_scalar('mse/train'            , average_mse_epoch    , epoch)

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            if models_weights_path == "None":
                model_name_save = "models_weights_{}/best_model_{}_weights.pth".format(model_name,model_name)
            else:
                model_name_save = "models_weights_{}/best__model_{}_weights.pth".format(model_name,model_name)
                model_name_save = os.path.join(models_weights_path,model_name_save)

            torch.save(train_model.model.state_dict(),model_name_save )

        # <-------------------------------------------------------------------->
        # <-------------------------------------------------------------------->

        ## VALIDATION STEP

        epoch_loss = 0
        #overall_acc_epoch, avg_per_class_acc_epoch, avg_jacc_epoch, avg_dice_epoch = 0, 0, 0, 0
        mIoU_acc_epoch, pixel_acc_epoch, average_mse_epoch = 0, 0, 0

        for top_batch, gts_batch in tqdm(valid_dataloader,desc = '[{}/{}] Evaluation'.format(epoch,epochs)):

            # reset batch metrics
            if vq_flag == True:
                batch_recon_error, batch_perplexity = 0, 0
                epoch_recon_error, epoch_perplexity = 0, 0

            #overall_acc_batch, avg_per_class_acc_batch, avg_jacc_batch, avg_dice_batch = 0, 0, 0, 0
            batch_mIoU_acc, batch_pixel_acc = 0, 0
            batch_loss = 0
            average_mse_batch = 0

            # moving batch data to device
            top_batch = top_batch.to(device)
            gts_batch = gts_batch.to(device)


            # evaluate batch
            if vq_flag == True:
                #batch_loss, batch_recon_error, batch_perplexity, average_mse_batch, metrics = evaluate_model(top_batch, gts_batch)

                batch_loss, batch_recon_error, batch_perplexity, average_mse_batch, batch_mIoU_acc, batch_pixel_acc = evaluate_model(top_batch, gts_batch)
                epoch_loss          += batch_loss
                epoch_recon_error   += batch_recon_error
                epoch_perplexity    += batch_perplexity
                average_mse_epoch   += average_mse_batch

                '''
                overall_acc_batch, avg_per_class_acc_batch, avg_jacc_batch, avg_dice_batch = metrics
                overall_acc_epoch       += overall_acc_batch
                avg_per_class_acc_epoch += avg_per_class_acc_batch
                avg_jacc_epoch          += avg_jacc_batch
                avg_dice_epoch          += avg_dice_batch
                '''

                mIoU_acc_epoch  += batch_mIoU_acc
                pixel_acc_epoch += batch_pixel_acc


            else:
                #batch_loss, average_mse_batch, metrics = evaluate_model(top_batch, gts_batch)

                batch_loss, average_mse_batch, batch_mIoU_acc, batch_pixel_acc = evaluate_model(top_batch, gts_batch)
                epoch_loss += batch_loss
                average_mse_epoch  += average_mse_batch

                '''overall_acc_batch, avg_per_class_acc_batch, avg_jacc_batch, avg_dice_batch = metrics
                overall_acc_epoch       += overall_acc_batch
                avg_per_class_acc_epoch += avg_per_class_acc_batch
                avg_jacc_epoch          += avg_jacc_batch
                avg_dice_epoch          += avg_dice_batch'''

                mIoU_acc_epoch  += batch_mIoU_acc
                pixel_acc_epoch += batch_pixel_acc

        # <-------------------------------------------------------------------->
        if vq_flag == True:
            epoch_recon_error   /= len(valid_dataloader)
            epoch_perplexity    /= len(valid_dataloader)

            print("\t reconstruction error : {:4f}".format(epoch_recon_error))
            logger.info("\t [epoch {}/{}] epoch reconstruction error : {}".format(epoch,epochs,epoch_recon_error))

            #print("\t Eval epoch perplexity : {}".format(epoch_perplexity))
            #logger.info("\t [epoch {}/{}] Eval epoch perplexity : {}".format(epoch,epochs,epoch_perplexity))

            # write on Tensorboard
            #writer.add_scalar('Perplexity/eval', epoch_perplexity, epoch)
            writer.add_scalar('ReconstructionError/eval', epoch_recon_error, epoch)

        epoch_loss              /= len(valid_dataloader)
        average_mse_epoch       /= len(valid_dataloader)

        mIoU_acc_epoch          /= len(valid_dataloader)
        pixel_acc_epoch         /= len(valid_dataloader)


        '''
        overall_acc_epoch       /= len(valid_dataloader)
        avg_per_class_acc_epoch /= len(valid_dataloader)
        avg_jacc_epoch          /= len(valid_dataloader)
        avg_dice_epoch          /= len(valid_dataloader)


        logger.info("\t [epoch {}/{}] average_mse_epoch : {}".format(epoch,epochs,average_mse_epoch))

        print("\t overall_acc_epoch : {}".format(overall_acc_epoch))
        print("\t avg_per_class_acc_epoch : {}".format(avg_per_class_acc_epoch))
        print("\t avg_jacc_epoch : {}".format(avg_jacc_epoch))
        print("\t avg_dice_epoch : {}".format(avg_dice_epoch))
        '''

        print("\t mIoU_acc_epoch     :  {:4f}".format(mIoU_acc_epoch))
        print("\t pixel_acc_epoch    :  {:4f}".format(pixel_acc_epoch))
        print("\t average_mse_epoch  :  {:4f}".format(average_mse_epoch))
        print("\t epoch loss         :  {:4f}".format(epoch_loss))

        print()
        logger.info("\t [epoch {}/{}] valid epoch loss : {}".format(epoch,epochs,epoch_loss))

        # <-------------------------------------------------------------------->

        # write on Tensorboard
        writer.add_scalar('Loss/eval'   , epoch_loss        , epoch)
        writer.add_scalar('mse/eval'    , average_mse_epoch , epoch)

        writer.add_scalar('mIoU/eval'       , mIoU_acc_epoch    , epoch)
        writer.add_scalar('pixel_acc/eval'  , pixel_acc_epoch   , epoch)

        # <-------------------------------------------------------------------->
        # <-------------------------------------------------------------------->


        # save models every 10 epochs
        if epoch % 25 == 0 and epoch != 0:
            print("\nsaving the models at epoch {}...".format(epoch))
            logger.info("\nsaving the models at epoch {}...".format(epoch))
            date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

            if models_weights_path == "None":
                model_name_save = "models_weights_{}/{}_model_weights_epoch_{}_{}.pth".format(model_name,model_name,epoch,date)
            else:
                model_name_save = "models_weights_{}/{}_model_weights_epoch_{}_{}.pth".format(model_name,model_name,epoch,date)
                model_name_save = os.path.join(models_weights_path,model_name_save)

            torch.save(train_model.model.state_dict(),model_name_save )
            print("models saved at epoch {}\n".format(epoch))
            logger.info("models saved at epoch {}".format(epoch))

        # <-------------------------------------------------------------------->
        # <-------------------------------------------------------------------->

    logger.info("train finished")
    print("\n\ntrain finished\n")

    writer.close()
    logger.info("writer closed")
    print("\nwriter closed")

if __name__ == "__main__":
    main()
