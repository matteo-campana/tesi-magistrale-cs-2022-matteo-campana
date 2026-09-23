import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class DINOLoss(nn.Module):

    """

    warmup_teacher_temp :   Initial value for the teacher temperature: 0.04 works
                            well in most cases. Try decreasing it if the training
                            loss does not decrease.

    teacher_temp :  Final value (after linear warmup) of the teacher temperature.
                    For most experiments, anything above 0.07 is unstable. We recommend starting
                    with the default value of 0.04 and increase this slightly if needed.

    warmup_teacher_temp_epochs :    Number of warmup epochs for the teacher
                                    temperature (Default: 30).

    """

    def __init__(self, device,
                 out_dim=0,
                 local_ncrops=6,
                 warmup_teacher_temp= 0.04,
                 teacher_temp= 0.04,
                 warmup_teacher_temp_epochs= 30,
                 nepochs=100,
                 student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.device = device
        self.student_temp = torch.tensor(student_temp)
        self.center_momentum = torch.tensor(center_momentum)
        self.local_ncrops = torch.tensor(local_ncrops)
        #self.register_buffer("center", torch.zeros(1,out_dim).to(device))
        self.center = torch.zeros(1,out_dim)
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp_schedule = torch.from_numpy(self.teacher_temp_schedule)


    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        #fix center size at first epoch
        if self.center.size(1) != teacher_output.view(1,-1).contiguous().size(1):
            self.center = torch.zeros(1,teacher_output.view(1,-1).contiguous().size(1))

        student_out = student_output / self.student_temp.to(self.device)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]

        teacher_out = F.softmax((teacher_output.view(1,-1).contiguous().to(self.device) - self.center.to(self.device)) / temp.to(self.device), dim=-1)

        teacher_out = teacher_out.view(teacher_output.size(0),teacher_output.size(1),-1).contiguous()
        student_out = student_out.view(student_output.size(0),student_output.size(1),-1).contiguous()

        total_loss = 0
        n_loss_terms = 0

        for index, t_batch in zip(range(teacher_out.size(0)),teacher_out):
            for global_crop in t_batch:

                q = global_crop.repeat(student_out.size(1),1).view(-1).contiguous()
                p = student_out[index].view(-1).contiguous()

                loss = torch.sum(-q.to(self.device) * F.log_softmax(p.to(self.device), dim=-1).to(self.device), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """

        teacher_output = teacher_output.view(1,-1).contiguous()

        #fix center size at first epoch
        if self.center.size(1) != teacher_output.size(1):
            self.center = torch.zeros(1,teacher_output.size(1))

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center.to(self.device) / len(teacher_output)

        # ema update
        self.center = self.center.to(self.device) * self.center_momentum.to(self.device) + batch_center.to(self.device) * (1 - self.center_momentum.to(self.device))
