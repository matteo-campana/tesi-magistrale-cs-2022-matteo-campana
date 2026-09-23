import torch
import torch.nn as nn
from tqdm import tqdm
import math
import torch.nn.functional as F


class EvaluateModel(nn.Module):

    def __init__(self,
                 teacher_model,
                 student_model,
                 vq_flag):
        super(EvaluateModel,self).__init__()

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.vq_flag = vq_flag


    @torch.no_grad()
    def forward(self,teacher_input,student_input):

        self.teacher_model.eval()
        self.student_model.eval()

        if self.vq_flag:
            _, _, _, _, teacher_output = self.teacher_model(teacher_input)
            vq_loss, quantized, perplexity, _, student_output = self.student_model(student_input)
        else:
            teacher_output = self.teacher_model(teacher_input)
            student_output = self.student_model(student_input)

        # flatten the output to compute the loss
        teacher_output = teacher_output.view(teacher_output.size(0),-1).contiguous()
        student_output = student_output.view(student_output.size(0),-1).contiguous()
        student_target = teacher_input.view(teacher_output.size(0),-1).contiguous()

        # compute the loss

        if self.vq_flag:
            data_variance = torch.var(student_input).item()
            recon_error = F.mse_loss( F.softmax(student_output), F.softmax(teacher_output)) / data_variance
            loss = recon_error + vq_loss
        else:
            loss = self.criterion(
                F.softmax(teacher_output),
                F.softmax(student_output)
                )

        average_mse_over_teacher    = F.mse_loss(student_output, teacher_output)
        average_mse_over_target     = F.mse_loss(student_output, student_target)

        if self.vq_flag:
            return loss.item() , recon_error.item(), perplexity.item(), average_mse_over_teacher.item() , average_mse_over_target.item()
        else:
            return loss.item(), average_mse_over_teacher.item() , average_mse_over_target.item()
