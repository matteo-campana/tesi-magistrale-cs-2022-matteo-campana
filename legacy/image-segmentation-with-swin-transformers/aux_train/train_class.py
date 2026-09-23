import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

'''
student = build_model(model_name="swin-transformer-vae")
teacher = build_model(model_name="swin-transformer-vae")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student.parameters(), lr=0.05, momentum=0.9)
batch_size = 8
dataloader = train_dataloader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

teacher = teacher.to(device)
student = student.to(device)

train_model = TrainModel(student_model = student,
                         teacher_model = teacher,
                         vq_flag = False,
                         criterion = criterion,
                         optimizer = optimizer,
                         alpha=0.99,
                         training = True)

epoch_loss = 0
batch_loss = 0
for batch in tqdm(dataloader):

    # split into teacher and student input,
    # the global crops goes to the teacher while the other goes to the student
    teacher_input = batch[0::2]
    student_input = batch[1::2]

    teacher_input = teacher_input.to(device)
    student_input = student_input.to(device)

    batch_loss = train_model(teacher_input,student_input)
    epoch_loss += batch_loss

epoch_loss /= len(dataloader)
print("\n epoch loss : {}".format(epoch_loss))

'''

class TrainModel(nn.Module):

    def __init__(self,
                 teacher_model,#= teacher ,
                 student_model,#= student ,
                 vq_flag,# = False,
                 criterion,# = criterion,
                 optimizer,# = optimizer,
                 alpha,#=0.99,
                 training):# = True):
        super(TrainModel,self).__init__()

        """
        alpha: the momentum coefficient between 0.0 and 1.0 for the teacher
            network updates. If alpha is a scalar (e.g., 0.99) then a static
            momentum coefficient is used during training. If alpha is tuple of
            two values, e.g., alpha=(alpha_base, num_iterations), then it
            uses a cosine schedule that starts from alpha_base and it increases
            it to 1.0 over num_iterations.
        """

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.vq_flag = vq_flag
        self.criterion = criterion
        self.optimizer = optimizer
        self.training = training

        # freeze the teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # check if the all the teacher's parameters are freezed
        for param in self.teacher_model.parameters():
            assert param.requires_grad == False


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


    def forward(self,teacher_input,student_input):

        # zero the gradient buffer
        self.optimizer.zero_grad()

        self.teacher_model.train()
        self.student_model.train()

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

        #update the student network since the teacher is freezed
        loss.backward()
        self.optimizer.step()

        #update the teacher
        self._update_teacher()

        if self.vq_flag:
            return loss.item() , recon_error.item(), perplexity.item(), average_mse_over_teacher.item() , average_mse_over_target.item()
        else:
            return loss.item(), average_mse_over_teacher.item() , average_mse_over_target.item()
