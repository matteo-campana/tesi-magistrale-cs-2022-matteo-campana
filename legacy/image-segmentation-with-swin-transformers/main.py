import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
#import torchvision.transforms.functional as F
import torchvision
from torchvision import transforms as T
import logging
import os
import numpy as np

from dataset import train_dataset
from models.build_model import build_model
from dino_augment_transforms import DataAugmentationDINO
from dataset.pretrain_dataset import dataset_pretrain, evaluation_dataset_pretrain
from aux_train.train_class import TrainModel
from aux_train.evaluate_class import EvaluateModel
import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import yaml


'''
    #Test messages
    logger.debug("Harmless debug Message")
    logger.info("Just an information")
    logger.warning("Its a Warning")
    logger.error("Did you try to divide by zero")
    logger.critical("Internet is down")
'''


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def main():

    '''model_name="swin-transformer-vq-vae"
    batch_size = 8
    epochs = 10
    dataset_length = 100
    train_data_path = "/content/drive/MyDrive/Colab Notebooks/semantic_labelling/Dataset/Vaihingen_processed/train"
    valid_data_path = "/content/drive/MyDrive/Colab Notebooks/semantic_labelling/Dataset/Vaihingen_processed/validation"
    learning_rate = 0.003'''



    with open('config/pretrain.yaml') as file:
        # scalar values to Python the dictionary format
        config = yaml.safe_load(file)
        #print(config)

    batch_size      = config["batch_size"]
    epochs          = config["epochs"]
    dataset_length  = config["dataset_length"]
    model_name      = config["model"]
    vq_flag         = config["vq_flag"]
    train_data_path = config["train_data_path"]
    valid_data_path = config["valid_data_path"]
    learning_rate   = config["learning_rate"]

    pretrained_model_student    = config["pretrained_model_student_name"]
    pretrained_model_teacher    = config["pretrained_model_teacher_name"]
    models_weights_path         = config["models_weights_path"]
    logdir_path                 = config["logdir_path"]
    last_epoch                  = config["last_epoch"]
    pretrain                    = config["pretrain"]


    assert not "ss" in model_name, "invalid model selected"

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

    global_crops_scale, local_crops_scale, local_crops_number = [0.14,1], [0.05, 0.14],   2,
    augment = DataAugmentationDINO(global_crops_scale, local_crops_scale, local_crops_number)

    transforms_dino = T.Compose([
            T.ToPILImage(),
            DataAugmentationDINO( [0.14,1], [0.05, 0.14],   2),
            T.Lambda(lambda crops: torch.stack([crop for crop in crops]))
        ])


    logger.info("initializing validation dataloader...")
    print("\ninitializing validation dataloader...")

    valid_dataset = evaluation_dataset_pretrain(data_path=valid_data_path,
                                size_w=224,
                                size_h=224,
                                batch_size=batch_size,
                                transform = None
                                )

    if dataset_length and dataset_length > 0:
        valid_dataloader = valid_dataset.data_iter_index(dataset_length)
    else:
        valid_dataloader = valid_dataset.data_iter()

    logger.info("validation dataloader initialized")
    print("validation dataloader initialized\n")

    logger.info("initializing train dataloader...")
    print("\ninitializing train dataloader...")

    train_dataset = dataset_pretrain(data_path=train_data_path,
                            size_w=224,
                            size_h=224,
                            batch_size=batch_size,
                            transform = transforms_dino
                            )


    if dataset_length and dataset_length > 0:
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

    student = build_model(model_name=model_name)
    teacher = build_model(model_name=model_name)

    last_epoch_model = last_epoch

    if pretrain and not pretrained_model_student == "None" and not pretrained_model_teacher == "None":

        student.load_state_dict(
            torch.load(
                os.path.join(
                    models_weights_path,
                    os.path.join("models_weights_{}".format(model_name),pretrained_model_student)
                    )
                )
            )

        teacher.load_state_dict(
            torch.load(
                os.path.join(
                    models_weights_path,
                    os.path.join("models_weights_{}".format(model_name),pretrained_model_teacher)
                    )
                )
            )

        #last_epoch_model = pretrained_model_student.replace("{}_student_model_weights_epoch_".format(model_name),"")
        last_epoch_model = pretrained_model_student.replace("{}_student_model_weights_epoch_".format(os.path.basename(model_name)),"")
        last_epoch_model = last_epoch_model[:last_epoch_model.find("_")]
        last_epoch_model = int(last_epoch_model)


    logger.info("model initialized")
    print("model initialized\n")

    # <------------------------------------------------------------------------>

    logger.info("initializing train arguments...")
    print("\ninitializing train arguments...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9)

    train_dataloader = train_dataloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vq_flag = True
    logger.info("vector quantize flag set to : {}".format(vq_flag))

    logger.info("train arguments initialized")
    print("train arguments initialized\n")

    # <------------------------------------------------------------------------>

    logger.info("moving models to device : {}...".format(device))
    print("\nmoving models to device : {}...".format(device))

    teacher = teacher.to(device)
    student = student.to(device)

    logger.info("models moved to device : {}".format(device))
    print("models moved to device : {}\n".format(device))

    # <------------------------------------------------------------------------>

    train_model = TrainModel(student_model = student,
                         teacher_model = teacher,
                         vq_flag = vq_flag,
                         criterion = criterion,
                         optimizer = optimizer,
                         alpha=0.99,
                         training = True)

    evaluate_model = EvaluateModel(teacher_model = student,
                                    student_model = teacher,
                                    vq_flag = vq_flag)



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

    for epoch in range(initial_epoch+1,epochs+1):

        # <-------------------------------------------------------------------->
        # <-------------------------------------------------------------------->
        # TRAIN STEP

        epoch_loss = 0
        batch_loss = 0
        batch_avg_mse_over_teacher , batch_avg_mse_over_target = 0, 0
        epoch_avg_mse_over_teacher , epoch_avg_mse_over_target = 0, 0

        if vq_flag == True:
            batch_recon_error, batch_perplexity = 0, 0
            epoch_recon_error, epoch_perplexity = 0, 0


        for teacher_input, student_input in tqdm(train_dataloader,desc = '[{}/{}] Training'.format(epoch,epochs)):

            # the global crops goes to the teacher while the other goes to the student
            teacher_input = teacher_input.to(device)
            student_input = student_input.to(device)

            if vq_flag == True:
                batch_loss, batch_recon_error, batch_perplexity, batch_avg_mse_over_teacher , batch_avg_mse_over_target = train_model(teacher_input,student_input)
                epoch_loss          += batch_loss
                epoch_recon_error   += batch_recon_error
                epoch_perplexity    += batch_perplexity

                epoch_avg_mse_over_teacher  += batch_avg_mse_over_teacher
                epoch_avg_mse_over_target   += batch_avg_mse_over_target

            else:
                batch_loss, batch_avg_mse_over_teacher , batch_avg_mse_over_target = train_model(teacher_input,student_input)
                epoch_loss += batch_loss

                epoch_avg_mse_over_teacher  += batch_avg_mse_over_teacher
                epoch_avg_mse_over_target   += batch_avg_mse_over_target

        # <-------------------------------------------------------------------->
        if vq_flag == True:
            epoch_recon_error   /= len(train_dataloader)
            epoch_perplexity    /= len(train_dataloader)

            print("\t epoch reconstruction error    : {}".format(epoch_recon_error))
            print("\t epoch perplexity              : {}".format(epoch_perplexity))

            logger.info("\t [epoch {}/{}] epoch reconstruction error    : {}".format(epoch,epochs,epoch_recon_error))
            logger.info("\t [epoch {}/{}] epoch perplexity              : {}".format(epoch,epochs,epoch_perplexity))

            # write on Tensorboard
            writer.add_scalar('Perplexity/train', epoch_perplexity, epoch)
            writer.add_scalar('ReconstructionError/train', epoch_recon_error, epoch)

        epoch_loss                  /= len(train_dataloader)
        epoch_avg_mse_over_teacher  /= len(train_dataloader)
        epoch_avg_mse_over_target   /= len(train_dataloader)

        print("\t epoch loss                : {}".format(epoch_loss))
        print("\t avg_mse_over_teacher      : {}".format(epoch_avg_mse_over_teacher))
        print("\t avg_mse_over_target       : {}".format(epoch_avg_mse_over_target))

        print()

        logger.info("\t [epoch {}/{}] epoch loss                    : {}".format(epoch,epochs,epoch_loss))
        logger.info("\t [epoch {}/{}] epoch_avg_mse_over_teacher    : {}".format(epoch,epochs,epoch_avg_mse_over_teacher))
        logger.info("\t [epoch {}/{}] epoch_avg_mse_over_target     : {}".format(epoch,epochs,epoch_avg_mse_over_target))

        # write on Tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('mse_over_teacher/train', epoch_avg_mse_over_teacher, epoch)
        writer.add_scalar('mse_over_target/train', epoch_avg_mse_over_target, epoch)

        # <-------------------------------------------------------------------->
        # <-------------------------------------------------------------------->

        ## VALIDATION STEP

        epoch_loss = 0
        batch_loss = 0
        batch_average_mse_over_teacher, batch_average_mse_over_target = 0, 0
        epoch_average_mse_over_teacher, epoch_average_mse_over_target = 0, 0

        if vq_flag == True:
            batch_recon_error, batch_perplexity = 0, 0
            epoch_recon_error, epoch_perplexity = 0, 0


        for input in tqdm(valid_dataloader,desc = '[{}/{}] Evaluation'.format(epoch,epochs)):
            # the global crops goes to the teacher while the other goes to the student
            teacher_input = input.to(device)
            student_input = input.to(device)

            if vq_flag == True:
                batch_loss, batch_recon_error, batch_perplexity, batch_average_mse_over_teacher, batch_average_mse_over_target = evaluate_model(teacher_input,student_input)
                epoch_loss          += batch_loss
                epoch_recon_error   += batch_recon_error
                epoch_perplexity    += batch_perplexity
                epoch_average_mse_over_teacher  += batch_average_mse_over_teacher
                epoch_average_mse_over_target   += batch_average_mse_over_target
            else:
                batch_loss, batch_average_mse_over_teacher, batch_average_mse_over_target = train_model(teacher_input,student_input)
                epoch_loss += batch_loss
                epoch_average_mse_over_teacher  += batch_average_mse_over_teacher
                epoch_average_mse_over_target   += batch_average_mse_over_target

        # <-------------------------------------------------------------------->
        if vq_flag == True:
            epoch_recon_error   /= len(valid_dataloader)
            epoch_perplexity    /= len(valid_dataloader)

            print("\t epoch reconstruction error   : {}".format(epoch_recon_error))
            print("\t epoch perplexity             : {}".format(epoch_perplexity))

            logger.info("\t [epoch {}/{}] Eval epoch reconstruction error : {}".format(epoch,epochs,epoch_recon_error))
            logger.info("\t [epoch {}/{}] Eval epoch perplexity : {}".format(epoch,epochs,epoch_perplexity))

            # write on Tensorboard
            writer.add_scalar('Perplexity/eval', epoch_perplexity, epoch)
            writer.add_scalar('ReconstructionError/eval', epoch_recon_error, epoch)


        epoch_loss                      /= len(valid_dataloader)
        epoch_average_mse_over_teacher  /= len(valid_dataloader)
        epoch_average_mse_over_target   /= len(valid_dataloader)

        logger.info("\t [epoch {}/{}] Eval epoch_loss    : {}".format(epoch,epochs,epoch_loss))
        logger.info("\t [epoch {}/{}] Eval epoch_average_mse_over_teacher    : {}".format(epoch,epochs,epoch_average_mse_over_teacher))
        logger.info("\t [epoch {}/{}] Eval epoch_average_mse_over_target     : {}".format(epoch,epochs,epoch_average_mse_over_target))

        print("\t valid epoch loss        : {}".format(epoch_loss))
        print("\t avg_mse_over_teacher    : {}".format(epoch_average_mse_over_teacher))
        print("\t avg_mse_over_target     : {}".format(epoch_average_mse_over_target))

        print()

        # write on Tensorboard
        writer.add_scalar('Loss/eval', epoch_loss, epoch)
        writer.add_scalar('mse_over_teacher/eval', epoch_average_mse_over_teacher, epoch)
        writer.add_scalar('mse_over_target/eval', epoch_average_mse_over_target, epoch)

        # <-------------------------------------------------------------------->


        # save models every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            print("\nsaving the models at epoch {}...".format(epoch))
            logger.info("\nsaving the models at epoch {}...".format(epoch))
            date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")


            if models_weights_path == "None":
                teacher_model_name = "models_weights_{}/{}_teacher_model_weights_epoch_{}_{}.pth".format(model_name,model_name,epoch,date)
                student_model_name = "models_weights_{}/{}_student_model_weights_epoch_{}_{}.pth".format(model_name,model_name,epoch,date)
            else:

                teacher_model_name = "models_weights_{}/{}_teacher_model_weights_epoch_{}_{}.pth".format(model_name,model_name,epoch,date)
                teacher_model_name = os.path.join(models_weights_path,teacher_model_name)

                student_model_name = "models_weights_{}/{}_student_model_weights_epoch_{}_{}.pth".format(model_name,model_name,epoch,date)
                student_model_name = os.path.join(models_weights_path,student_model_name)

            torch.save(train_model.teacher_model.state_dict(),teacher_model_name )
            torch.save(train_model.student_model.state_dict(),student_model_name )
            print("models saved at epoch {}\n".format(epoch))
            logger.info("models saved at epoch {}".format(epoch))

        # <-------------------------------------------------------------------->

    logger.info("train finished")
    print("\n\ntrain finished\n")

    writer.close()
    logger.info("writer closed")
    print("\nwriter closed")


if __name__ == "__main__":
    main()
