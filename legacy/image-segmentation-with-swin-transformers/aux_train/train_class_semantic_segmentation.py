import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F
from metrics import mIoU, pixel_accuracy

class TrainModelSS(nn.Module):

    def __init__(self,
                 model,
                 vq_flag,# = False,
                 criterion,# = criterion,
                 optimizer,# = optimizer,
                 training):# = True):
        super(TrainModelSS,self).__init__()

        self.model = model
        self.vq_flag = vq_flag
        self.criterion = criterion
        self.optimizer = optimizer
        self.training = training

    def forward(self,x,target):

        # zero the gradient buffer
        self.optimizer.zero_grad()

        self.model.train()

        if self.vq_flag:
            vq_loss, quantized, perplexity, _, output = self.model(x)
        else:
            output = self.model(x)

        mIoU_acc    = mIoU(output, target, smooth=1e-10, n_classes=6)
        pixel_acc   = pixel_accuracy(output, target)

        # flatten the output to compute the loss
        # output = output.view(output.size(0),-1).contiguous()
        # target = target.view(target.size(0),-1).contiguous()

        average_mse = F.mse_loss(output, target)

        # compute the loss

        if self.vq_flag:
            out = torch.argmax(output,dim=1).view(output.size(0),1,output.size(2),output.size(3)).float()
            data_variance = torch.var(out).item()
            recon_error = F.mse_loss( F.softmax(out,dim=1), target) / data_variance
            loss = recon_error + vq_loss
        else:
            loss = self.criterion(output, target.squeeze(dim = 1).long())



        loss.backward()
        self.optimizer.step()

        if self.vq_flag:
            return loss.item() , recon_error.item(), perplexity.item(), mIoU_acc, pixel_acc, average_mse
        else:
            return loss.item(), mIoU_acc, pixel_acc, average_mse
