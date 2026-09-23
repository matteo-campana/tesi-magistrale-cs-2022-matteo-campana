import os
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from metrics import mIoU, pixel_accuracy
from models.build_model import build_model
from dataset.dataset_features_extractor import FeaturesExtractorDataset
from torch.utils.data import DataLoader
from utils.dataset_utils import from_one_class_to_rgb_jit
from torchvision.utils import make_grid
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import math
from aux_train.dino_loss import DINOLoss

from dataset.dataset_potsdam_features_extractor import DatasetFeaturesExtractorPotsdam

class LitFeaturesExtractor(pl.LightningModule):
    def __init__(self, model_name, vq_flag,
                 train_data_path, valid_data_path,
                 learning_rate = 1e-6, batch_size = 8,
                 train_dataset_length=-1,valid_dataset_length=-1,
                 local_crops_number = 6,
                 nepochs=100,
                 dataset_name = "Vaihingen"):
        super().__init__()

        self.model_name = model_name

        self.teacher_model = build_model(model_name = model_name)
        self.student_model = build_model(model_name = model_name)

        # freeze the teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # check if the all the teacher's parameters are freezed
        for param in self.teacher_model.parameters():
            assert param.requires_grad == False

        self.vq_flag = vq_flag

        self.best_loss_train = 0
        self.best_loss_val = 0

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.hparams.batch_size = batch_size
        self.hparams.learning_rate = learning_rate

        self.train_dataset_length, self.valid_dataset_length = train_dataset_length, valid_dataset_length

        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path

        self.local_crops_number = local_crops_number
        self.nepochs = nepochs

        out_dim = torch.rand(self.hparams.batch_size,2,3,224,224).view(1,-1).contiguous().size(1)
        # ============ preparing loss ... ============
        self.dino_loss = DINOLoss(
                device=self.device,
                out_dim=out_dim,
                local_ncrops=self.local_crops_number,
                warmup_teacher_temp = 0.04,
                teacher_temp = 0.04,
                warmup_teacher_temp_epochs = 0,
                nepochs = self.nepochs,
                student_temp = 0.1,
                center_momentum=0.9
        )

        self.best_loss_train, self.best_loss_val = 0, 0

        self.dataset_name = dataset_name

        #alpha = (0.996 , self.len_train_dataloader)

        alpha = 0.996

        if isinstance(alpha, (tuple, list)):
            # Use cosine schedule in order to increase the alpha from
            # alpha_base (e.g., 0.99) to 1.0.
            alpha_base, num_iterations = alpha
            self._alpha_base = alpha_base
            self._num_iterations = num_iterations
            self.register_buffer("_alpha", torch.FloatTensor(1).fill_(alpha_base))
            self.register_buffer("_iteration", torch.zeros(1))
            self._alpha_cosine_schedule = True
        else:
            self._alpha = alpha
            self._alpha_cosine_schedule = False


    @torch.no_grad()
    def _get_momentum_alpha(self):
        if self._alpha_cosine_schedule:
            scale = 0.5 * (1. + math.cos((math.pi * self._iteration.item()) / self._num_iterations))
            self._alpha.fill_(1.0 - (1.0 - self._alpha_base) * scale)
            self._iteration += 1
            return self._alpha.item()
        else:
            return self._alpha

    @torch.no_grad()
    def _update_teacher(self):
        """ Exponetial moving average for the teacher params:
            param_teacher = param_teacher * alpha + param * (1-alpha)
        """
        if not self.training:
            return
        alpha = self._get_momentum_alpha()
        if alpha >= 1.0:
            return
        for param_student, param_teacher in zip(self.student_model.parameters(), self.teacher_model.parameters() ):
            param_teacher.data.mul_(alpha).add_(param_student.detach().data, alpha=(1. - alpha))

    def forward(self, x):
        x = self.student(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.learning_rate)

        return optimizer


    def training_step(self, train_batch, batch_idx):

        global_crops_batch, local_crops_batch = train_batch

        global_crops_batch  = global_crops_batch.view(-1, global_crops_batch.size(2), global_crops_batch.size(3), global_crops_batch.size(4)).contiguous()
        local_crops_batch   = local_crops_batch.view(-1, local_crops_batch.size(2), local_crops_batch.size(3), local_crops_batch.size(4)).contiguous()

        if self.vq_flag:
            vq_loss_teacher, quantized_teacher, perplexity_teacher, encodings_teacher, teacher_out = self.teacher_model(global_crops_batch)
            vq_loss_student, quantized_student, perplexity_student, encodings_student, student_out = self.student_model(local_crops_batch)
        else:
            teacher_out = self.teacher_model(global_crops_batch)
            student_out = self.student_model(local_crops_batch)

        teacher_out = teacher_out.view(-1, 2, teacher_out.size(1), teacher_out.size(2), teacher_out.size(3) ).contiguous()
        student_out = student_out.view(-1, self.local_crops_number, student_out.size(1), student_out.size(2), student_out.size(3) ).contiguous()

        loss = self.dino_loss(student_out, teacher_out, self.current_epoch)

        #update teacher network
        self._update_teacher()

        if self.vq_flag:
            step_dictionary = {
                "loss"  : loss,
                "perplexity_teacher" : perplexity_teacher,
                "perplexity_student" : perplexity_student,
                "vq_loss_teacher" : vq_loss_teacher,
                "vq_loss_student" : vq_loss_student
            }
        else:
            step_dictionary = {
                "loss"  : loss,
            }

        return step_dictionary


    def training_epoch_end(self, train_step_outputs):
        # calculating average metrics
        avg_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()

        # write average metrics
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

        if self.vq_flag:
            avg_perplexity_teacher  = torch.stack([x['perplexity_teacher']   for x in train_step_outputs]).mean()
            avg_perplexity_student  = torch.stack([x['perplexity_student']   for x in train_step_outputs]).mean()

            avg_vq_loss_teacher  = torch.stack([x['vq_loss_teacher']   for x in train_step_outputs]).mean()
            avg_vq_loss_student  = torch.stack([x['vq_loss_student']   for x in train_step_outputs]).mean()

            self.logger.experiment.add_scalar("perplexity_teacher/Train", avg_perplexity_teacher, self.current_epoch)
            self.logger.experiment.add_scalar("perplexity_student/Train", avg_perplexity_student, self.current_epoch)

            self.logger.experiment.add_scalar("vq_loss_teacher/Train", avg_vq_loss_teacher, self.current_epoch)
            self.logger.experiment.add_scalar("vq_loss_student/Train", avg_vq_loss_student, self.current_epoch)

        '''if self.best_loss_val > avg_loss or self.current_epoch == 0:
            self.best_loss_val = avg_loss

        self.logger.experiment.add_scalar("best_loss/Train", self.best_loss_train, self.current_epoch)'''




    def validation_step(self, val_batch, batch_idx):
        global_crops_batch, local_crops_batch = val_batch

        global_crops_batch  = global_crops_batch.view(-1, global_crops_batch.size(2), global_crops_batch.size(3), global_crops_batch.size(4)).contiguous()
        local_crops_batch   = local_crops_batch.view(-1, local_crops_batch.size(2), local_crops_batch.size(3), local_crops_batch.size(4)).contiguous()

        if self.vq_flag:
            vq_loss_teacher, quantized_teacher, perplexity_teacher, encodings_teacher, teacher_out = self.teacher_model(global_crops_batch)
            vq_loss_student, quantized_student, perplexity_student, encodings_student, student_out = self.student_model(local_crops_batch)
        else:
            teacher_out = self.teacher_model(global_crops_batch)
            student_out = self.student_model(local_crops_batch)

        teacher_out = teacher_out.view(-1, 2, teacher_out.size(1), teacher_out.size(2), teacher_out.size(3) ).contiguous()
        student_out = student_out.view(-1, self.local_crops_number, student_out.size(1), student_out.size(2), student_out.size(3) ).contiguous()

        loss = self.dino_loss(student_out, teacher_out, self.current_epoch)

        if self.vq_flag:
            step_dictionary = {
                "loss"  : loss,
                "perplexity_teacher" : perplexity_teacher,
                "perplexity_student" : perplexity_student,
                "vq_loss_teacher" : vq_loss_teacher,
                "vq_loss_student" : vq_loss_student
            }
        else:
            step_dictionary = {
                "loss"  : loss,
            }

        return step_dictionary


    def validation_epoch_end(self, validation_step_outputs):
        # calculating average metrics
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()

        # write average metrics
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)

        if self.vq_flag:
            avg_perplexity_teacher  = torch.stack([x['perplexity_teacher']   for x in validation_step_outputs]).mean()
            avg_perplexity_student  = torch.stack([x['perplexity_student']   for x in validation_step_outputs]).mean()

            avg_vq_loss_teacher  = torch.stack([x['vq_loss_teacher']   for x in validation_step_outputs]).mean()
            avg_vq_loss_student  = torch.stack([x['vq_loss_student']   for x in validation_step_outputs]).mean()

            self.logger.experiment.add_scalar("perplexity_teacher/Val", avg_perplexity_teacher, self.current_epoch)
            self.logger.experiment.add_scalar("perplexity_student/Val", avg_perplexity_student, self.current_epoch)

            self.logger.experiment.add_scalar("vq_loss_teacher/Val", avg_vq_loss_teacher, self.current_epoch)
            self.logger.experiment.add_scalar("vq_loss_student/Val", avg_vq_loss_student, self.current_epoch)

        '''if self.best_loss_val > avg_loss or self.current_epoch == 0:
            self.best_loss_val = avg_loss

        self.logger.experiment.add_scalar("best_loss/Val", self.best_loss_val, self.current_epoch)'''


    def train_dataloader(self):

        assert self.dataset_name == "Vaihingen" or self.dataset_name == "Potsdam", "[ERROR] Wrong dataset selected..."
        if self.dataset_name == "Vaihingen":
            train_dataset = FeaturesExtractorDataset(   data_path=self.train_data_path,
                                                        size_w=224,
                                                        size_h=224,
                                                        dataset_length=self.train_dataset_length,
                                                        global_crops_scale  = [0.14,1],
                                                        local_crops_scale   = [0.05, 0.14],
                                                        local_crops_number  = self.local_crops_number
                                                    )

            dataloader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size, shuffle=True)
            self.len_train_dataloader = len(dataloader)
        elif self.dataset_name == "Potsdam":
            train_dataset = DatasetFeaturesExtractorPotsdam(   data_path=self.train_data_path,
                                                        size_w=224,
                                                        size_h=224,
                                                        dataset_length=self.train_dataset_length,
                                                        global_crops_scale  = [0.14,1],
                                                        local_crops_scale   = [0.05, 0.14],
                                                        local_crops_number  = self.local_crops_number
                                                    )

            dataloader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size, shuffle=True)
            self.len_train_dataloader = len(dataloader)


        out_dim = torch.rand(self.hparams.batch_size,2,3,224,224).view(1,-1).contiguous().size(1)
        # ============ preparing loss ... ============
        self.dino_loss = DINOLoss(
                device=self.device,
                out_dim=out_dim,
                local_ncrops=self.local_crops_number,
                warmup_teacher_temp = 0.04,
                teacher_temp = 0.04,
                warmup_teacher_temp_epochs = 0,
                nepochs = self.nepochs,
                student_temp = 0.1,
                center_momentum=0.9
        )

        alpha = (0.996 , self.len_train_dataloader)

        #alpha = 0.996

        if isinstance(alpha, (tuple, list)):
            # Use cosine schedule in order to increase the alpha from
            # alpha_base (e.g., 0.99) to 1.0.
            alpha_base, num_iterations = alpha
            self._alpha_base = alpha_base
            self._num_iterations = num_iterations
            self._alpha = torch.FloatTensor(1).fill_(alpha_base)
            self._iteration = torch.zeros(1)
            self._alpha_cosine_schedule = True
        else:
            self._alpha = alpha
            self._alpha_cosine_schedule = False

        return dataloader

    def val_dataloader(self):
        assert self.dataset_name == "Vaihingen" or self.dataset_name == "Potsdam", "[ERROR] Wrong dataset selected..."
        if self.dataset_name == "Vaihingen":
            valid_dataset = FeaturesExtractorDataset(   data_path=self.valid_data_path,
                                                        size_w=224,
                                                        size_h=224,
                                                        dataset_length=self.valid_dataset_length,
                                                        global_crops_scale  = [0.14,1],
                                                        local_crops_scale   = [0.05, 0.14],
                                                        local_crops_number  = self.local_crops_number
                                                    )

            dataloader =  DataLoader(dataset=valid_dataset, batch_size=self.hparams.batch_size, shuffle=True)
            self.len_val_dataloader = len(dataloader)
            return dataloader
        elif self.dataset_name == "Potsdam":
            valid_dataset = DatasetFeaturesExtractorPotsdam(   data_path=self.valid_data_path,
                                                        size_w=224,
                                                        size_h=224,
                                                        dataset_length=self.valid_dataset_length,
                                                        global_crops_scale  = [0.14,1],
                                                        local_crops_scale   = [0.05, 0.14],
                                                        local_crops_number  = self.local_crops_number
                                                    )

            dataloader =  DataLoader(dataset=valid_dataset, batch_size=self.hparams.batch_size, shuffle=True)
            self.len_val_dataloader = len(dataloader)
            return dataloader
