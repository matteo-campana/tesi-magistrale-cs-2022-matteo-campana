import os
import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from utils.dataset_utils import from_one_class_to_rgb_jit
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import numpy as np
from dataset.dataset_semantic_segmentation import Dataset_Semantic_Segmentation
from dataset.dataset_potsdam_semantic_segmentation import DatasetPotsdamSemantiSegmentatin

from metrics import mIoU, pixel_accuracy, fast_hist, eval_metrics
from models.build_model import build_model


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model_name, vq_flag, ref_im, ref_gt,
                 train_data_path, valid_data_path,
                 learning_rate = 1e-6, batch_size = 8,
                 train_dataset_length=-1,valid_dataset_length=-1,
                 dataset_name = "Potsdam"):
        super().__init__()

        self.model_name = model_name
        self.model = build_model(model_name = model_name)
        self.criterion = nn.CrossEntropyLoss( ignore_index = 0 )

        self.vq_flag = vq_flag

        self.best_loss_train = 0
        self.best_loss_val = 0

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.hparams.batch_size = batch_size
        self.hparams.learning_rate = learning_rate

        self.reference_image, self.reference_gt = ref_im, ref_gt
        self.train_dataset_length, self.valid_dataset_length = train_dataset_length, valid_dataset_length

        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path

        self.dataset_name = dataset_name

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.learning_rate)
        return optimizer


    def training_step(self, train_batch, batch_idx):

        x, target = train_batch

        if self.vq_flag:
            vq_loss, quantized, perplexity, encodings, output = self.model(x)
        else:
            output = self.model(x)

        mIoU_acc    = mIoU(output, target, smooth=1e-10, n_classes=6)
        #pixel_acc   = pixel_accuracy(output, target)
        mse_acc     = F.mse_loss(output, target)

        true = target
        pred = output
        pred = torch.argmax(F.softmax(pred, dim=1),dim=1).view(pred.size(0),1,pred.size(2),pred.size(3))
        true = torch.squeeze(true.cpu(), 0)
        pred = torch.squeeze(pred.cpu(), 0)
        hist = fast_hist(true.flatten().type(torch.LongTensor), pred.flatten().type(torch.LongTensor), 6)
        overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(hist, verbose = False)

        mIoU_acc = torch.tensor(mIoU_acc)
        #pixel_acc = torch.tensor(pixel_acc)

        overall_acc, avg_per_class_acc, avg_jacc, avg_dice = torch.tensor(overall_acc), torch.tensor(avg_per_class_acc), torch.tensor(avg_jacc), torch.tensor(avg_dice)

        if self.vq_flag:
            cross_entrpy_loss = self.criterion(output, target.squeeze(dim = 1).long())
            reluctance_to_change_threshold = cross_entrpy_loss*10
            if vq_loss > reluctance_to_change_threshold:
                vq_loss_component = reluctance_to_change_threshold
            elif vq_loss < -reluctance_to_change_threshold:
                vq_loss_component = -reluctance_to_change_threshold
            else:
                vq_loss_component = vq_loss
            loss = cross_entrpy_loss + vq_loss_component
        else:
            loss = self.criterion(output, target.squeeze(dim = 1).long())

        if self.vq_flag:
            step_dictionary = {
                "loss"  : loss,
                "mse"   : mse_acc,
                "mIoU"  : mIoU_acc,
                #"pixel_acc"     : pixel_acc,
                "perplexity"    : perplexity,
                "cross_entrpy_loss" : cross_entrpy_loss,
                "vq_loss" : vq_loss,
                "overall_acc" :overall_acc,
                "avg_per_class_acc":avg_per_class_acc,
                "avg_jacc":avg_jacc,
                "avg_dice":avg_dice
            }
        else:
            step_dictionary = {
                "loss"  : loss,
                "mse"   : mse_acc,
                "mIoU"  : mIoU_acc,
                #"pixel_acc" : pixel_acc
                "overall_acc" :overall_acc,
                "avg_per_class_acc":avg_per_class_acc,
                "avg_jacc":avg_jacc,
                "avg_dice":avg_dice
            }

        return step_dictionary


    def training_epoch_end(self, train_step_outputs):
		#  the function is called after every epoch is completed
        if(self.current_epoch==0):
            sampleImg = torch.rand( (self.hparams.batch_size,self.reference_image.shape[1],self.reference_image.shape[2],self.reference_image.shape[3]) , device=self.device)
            self.logger.experiment.add_graph( self ,sampleImg)

        # calculating average metrics
        avg_loss        = torch.stack([x['loss']        for x in train_step_outputs]).mean()
        avg_mse         = torch.stack([x['mse']         for x in train_step_outputs]).mean()
        avg_mIoU        = torch.stack([x['mIoU']        for x in train_step_outputs]).mean()
        #avg_pixel_acc   = torch.stack([x['pixel_acc']   for x in train_step_outputs]).mean()

        overall_acc         = torch.stack([x['overall_acc']         for x in train_step_outputs]).mean()
        avg_per_class_acc   = torch.stack([x['avg_per_class_acc']   for x in train_step_outputs]).mean()
        avg_jacc            = torch.stack([x['avg_jacc']            for x in train_step_outputs]).mean()
        avg_dice            = torch.stack([x['avg_dice']            for x in train_step_outputs]).mean()

        # write average metrics
        self.logger.experiment.add_scalar("Loss/Train"      , avg_loss       , self.current_epoch)
        self.logger.experiment.add_scalar("mse/Train"       , avg_mse        , self.current_epoch)
        self.logger.experiment.add_scalar("mIoU/Train"      , avg_mIoU       , self.current_epoch)
        #self.logger.experiment.add_scalar("pixel_acc/Train" , avg_pixel_acc  , self.current_epoch)

        self.logger.experiment.add_scalar("overall_acc/Train"       , overall_acc       , self.current_epoch)
        self.logger.experiment.add_scalar("avg_per_class_acc/Train" , avg_per_class_acc , self.current_epoch)
        self.logger.experiment.add_scalar("avg_jacc/Train"          , avg_jacc          , self.current_epoch)
        self.logger.experiment.add_scalar("avg_dice/Train"          , avg_dice          , self.current_epoch)

        if self.vq_flag:
            avg_perplexity          = torch.stack([x['perplexity']          for x in train_step_outputs]).mean()
            avg_cross_entrpy_loss   = torch.stack([x['cross_entrpy_loss']   for x in train_step_outputs]).mean()
            avg_vq_loss             = torch.stack([x['vq_loss']             for x in train_step_outputs]).mean()

            self.logger.experiment.add_scalar("perplexity/Train"          , avg_perplexity        , self.current_epoch)
            self.logger.experiment.add_scalar("cross_entrpy_loss/Train"   , avg_cross_entrpy_loss , self.current_epoch)
            self.logger.experiment.add_scalar("vq_loss/Train"             , avg_vq_loss           , self.current_epoch)

        if self.best_loss_train > avg_loss or self.current_epoch == 0:
            self.best_loss_train = avg_loss
        self.logger.experiment.add_scalar("best_loss/Train", self.best_loss_train, self.current_epoch)

        # logging histograms
        #for name,params in self.named_parameters():
        #    self.logger.experiment.add_histogram(name,params,self.current_epoch)

        # add ref images every 10 epochs
        self._log_images()


    def _log_images(self):
        # add ref images every 10 epochs
        if((self.current_epoch + 1) % 10 ):

            if self.vq_flag:
                _, _, _, _, pred = self.model(self.reference_image.to(self.device))
                pred = pred.detach().cpu()
            else:
                pred = self.model(self.reference_image.to(self.device)).detach().cpu()

            pred = torch.argmax(pred,dim=1).view(pred.size(0),1,pred.size(2),pred.size(3)).float()

            gt_3c   = torch.tensor(np.array(list(map(lambda x : from_one_class_to_rgb_jit(x.float().numpy()),self.reference_gt))))
            pred_3c = torch.tensor(np.array(list(map(lambda x : from_one_class_to_rgb_jit(x.float().numpy()),pred))))

            grid_gt_3c              = make_grid(gt_3c,                  nrow = gt_3c.shape[0])
            grid_pred_3c            = make_grid(pred_3c,                nrow = pred_3c.shape[0])
            grid_reference_image    = make_grid(self.reference_image,   nrow = self.reference_image.shape[0])

            self.logger.experiment.add_image("input"        , grid_reference_image  , self.current_epoch, dataformats="CHW")
            self.logger.experiment.add_image("ground truth" , grid_gt_3c            , self.current_epoch, dataformats="CHW")
            self.logger.experiment.add_image("predictions"  , grid_pred_3c          , self.current_epoch, dataformats="CHW")



    def validation_step(self, val_batch, batch_idx):

        x, target = val_batch

        if self.vq_flag:
            vq_loss, quantized, perplexity, encodings, output = self.model(x)
        else:
            output = self.model(x)

        mIoU_acc    = mIoU(output, target, smooth=1e-10, n_classes=6)
        #pixel_acc   = pixel_accuracy(output, target)
        mse_acc     = F.mse_loss(output, target)

        true = target
        pred = output
        pred = torch.argmax(F.softmax(pred, dim=1),dim=1).view(pred.size(0),1,pred.size(2),pred.size(3))
        true = torch.squeeze(true.cpu(), 0)
        pred = torch.squeeze(pred.cpu(), 0)
        hist = fast_hist(true.flatten().type(torch.LongTensor), pred.flatten().type(torch.LongTensor), 6)
        overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(hist, verbose = False)

        mIoU_acc = torch.tensor(mIoU_acc)
        #pixel_acc = torch.tensor(pixel_acc)

        overall_acc, avg_per_class_acc, avg_jacc, avg_dice = torch.tensor(overall_acc), torch.tensor(avg_per_class_acc), torch.tensor(avg_jacc), torch.tensor(avg_dice)

        if self.vq_flag:
            cross_entrpy_loss = self.criterion(output, target.squeeze(dim = 1).long())
            reluctance_to_change_threshold = cross_entrpy_loss*10
            if vq_loss > reluctance_to_change_threshold:
                vq_loss_component = reluctance_to_change_threshold
            elif vq_loss < -reluctance_to_change_threshold:
                vq_loss_component = -reluctance_to_change_threshold
            else:
                vq_loss_component = vq_loss
            loss = cross_entrpy_loss + vq_loss_component
        else:
            loss = self.criterion(output, target.squeeze(dim = 1).long())

        if self.vq_flag:
            step_dictionary = {
                "loss"  : loss,
                "mse"   : mse_acc,
                "mIoU"  : mIoU_acc,
                #"pixel_acc"     : pixel_acc,
                "perplexity"    : perplexity,
                "cross_entrpy_loss" : cross_entrpy_loss,
                "vq_loss" : vq_loss,
                "overall_acc" :overall_acc,
                "avg_per_class_acc":avg_per_class_acc,
                "avg_jacc":avg_jacc,
                "avg_dice":avg_dice
            }
        else:
            step_dictionary = {
                "loss"  : loss,
                "mse"   : mse_acc,
                "mIoU"  : mIoU_acc,
                #"pixel_acc" : pixel_acc
                "overall_acc" :overall_acc,
                "avg_per_class_acc":avg_per_class_acc,
                "avg_jacc":avg_jacc,
                "avg_dice":avg_dice
            }

        return step_dictionary


    def validation_epoch_end(self, validation_step_outputs):
        # calculating average metrics
        avg_loss        = torch.stack([x['loss']        for x in validation_step_outputs]).mean()
        avg_mse         = torch.stack([x['mse']         for x in validation_step_outputs]).mean()
        avg_mIoU        = torch.stack([x['mIoU']        for x in validation_step_outputs]).mean()
        #avg_pixel_acc   = torch.stack([x['pixel_acc']   for x in validation_step_outputs]).mean()

        overall_acc         = torch.stack([x['overall_acc']         for x in validation_step_outputs]).mean()
        avg_per_class_acc   = torch.stack([x['avg_per_class_acc']   for x in validation_step_outputs]).mean()
        avg_jacc            = torch.stack([x['avg_jacc']            for x in validation_step_outputs]).mean()
        avg_dice            = torch.stack([x['avg_dice']            for x in validation_step_outputs]).mean()

        # write average metrics
        self.logger.experiment.add_scalar("Loss/Val"      , avg_loss       , self.current_epoch)
        self.logger.experiment.add_scalar("mse/Val"       , avg_mse        , self.current_epoch)
        self.logger.experiment.add_scalar("mIoU/Val"      , avg_mIoU       , self.current_epoch)
        #self.logger.experiment.add_scalar("pixel_acc/Val" , avg_pixel_acc  , self.current_epoch)

        self.logger.experiment.add_scalar("overall_acc/Val"       , overall_acc       , self.current_epoch)
        self.logger.experiment.add_scalar("avg_per_class_acc/Val" , avg_per_class_acc , self.current_epoch)
        self.logger.experiment.add_scalar("avg_jacc/Val"          , avg_jacc          , self.current_epoch)
        self.logger.experiment.add_scalar("avg_dice/Val"          , avg_dice          , self.current_epoch)

        if self.vq_flag:
            avg_perplexity          = torch.stack([x['perplexity']          for x in validation_step_outputs]).mean()
            avg_cross_entrpy_loss   = torch.stack([x['cross_entrpy_loss']   for x in validation_step_outputs]).mean()
            avg_vq_loss             = torch.stack([x['vq_loss']             for x in validation_step_outputs]).mean()

            self.logger.experiment.add_scalar("perplexity/Val"          , avg_perplexity        , self.current_epoch)
            self.logger.experiment.add_scalar("cross_entrpy_loss/Val"   , avg_cross_entrpy_loss , self.current_epoch)
            self.logger.experiment.add_scalar("vq_loss/Val"             , avg_vq_loss           , self.current_epoch)


        if self.best_loss_val > avg_loss or self.current_epoch == 0:
            self.best_loss_val = avg_loss
        self.logger.experiment.add_scalar("best_loss/Val", self.best_loss_val, self.current_epoch)

    def train_dataloader(self):
        assert self.dataset_name == "Vaihingen" or self.dataset_name == "Potsdam", "[ERROR] Wrong dataset selected..."
        if self.dataset_name == "Vaihingen":
            train_dataset = Dataset_Semantic_Segmentation(  data_path=self.train_data_path,
                                                    size_w=224,
                                                    size_h=224,
                                                    batch_size=self.hparams.batch_size,
                                                    transform = None,
                                                    dataset_length=self.train_dataset_length
                                                )
            return DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size)
        elif self.dataset_name == "Potsdam":
            train_dataset = DatasetPotsdamSemantiSegmentatin( data_path=self.train_data_path,
                                                    size_w=224,
                                                    size_h=224,
                                                    transform = None,
                                                    dataset_length=self.train_dataset_length
                                                    )
            return DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        assert self.dataset_name == "Vaihingen" or self.dataset_name == "Potsdam", "[ERROR] Wrong dataset selected..."
        if self.dataset_name == "Vaihingen":
            valid_dataset = Dataset_Semantic_Segmentation(  data_path=self.valid_data_path,
                                                        size_w=224,
                                                        size_h=224,
                                                        batch_size=self.hparams.batch_size,
                                                        transform = None,
                                                        dataset_length=self.valid_dataset_length
                                                        )
            return DataLoader(dataset=valid_dataset, batch_size=self.hparams.batch_size)
        elif self.dataset_name == "Potsdam":
            valid_dataset = DatasetPotsdamSemantiSegmentatin( data_path=self.valid_data_path,
                                                    size_w=224,
                                                    size_h=224,
                                                    transform = None,
                                                    dataset_length=self.train_dataset_length
                                                    )
            return DataLoader(dataset=valid_dataset, batch_size=self.hparams.batch_size)
