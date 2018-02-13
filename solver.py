import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from dataset import *
import misc

from model import Generator, Discriminator


class Solver(object):
    def __init__(self, config):
        self.cuda = torch.cuda.is_available()

        # model
        self.G_A = None
        self.G_B = None
        self.D_A = None
        self.D_B = None

        # criterion/optimizer
        self.MSE_loss = None
        self.L1_loss = None
        self.G_optimizer = None
        self.D_A_optimizer = None
        self.D_B_optimizer = None

        # dataloader
        self.dataset = config.dataset
        self.train_data_loader_A = None
        self.train_data_loader_B = None
        self.test_data_loader_A = None
        self.test_data_loader_B = None

        # hyper-parameters
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.num_resnet = config.num_resnet
        self.num_epochs = config.num_epochs
        self.decay_epoch = config.decay_epoch
        self.lrG = config.lrG
        self.lrD = config.lrD
        self.lambdaA = config.lambdaA
        self.lambdaB = config.lambdaB
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # dataset/dataloader
        self.resize_scale = config.resize_scale
        self.crop_size = config.crop_size
        self.flip = config.flip
        self.model_dir = './' + self.dataset + '/models/'

    def build_model(self):

        self.G_A = Generator(3, self.g_conv_dim, 3, self.num_resnet)
        self.G_B = Generator(3, self.g_conv_dim, 3, self.num_resnet)
        self.D_A = Discriminator(3, self.d_conv_dim, 1)
        self.D_B = Discriminator(3, self.d_conv_dim, 1)

        self.G_A.normal_weight_init(mean=0.0, std=0.02)
        self.G_B.normal_weight_init(mean=0.0, std=0.02)
        self.D_A.normal_weight_init(mean=0.0, std=0.02)
        self.D_B.normal_weight_init(mean=0.0, std=0.02)

        self.MSE_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()

        self.G_optimizer = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_A_optimizer = torch.optim.Adam(self.D_A.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))
        self.D_B_optimizer = torch.optim.Adam(self.D_B.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

        if self.cuda:
            self.G_A.cuda()
            self.G_B.cuda()
            self.D_A.cuda()
            self.D_B.cuda()
            cudnn.benchmark = True

            self.MSE_loss.cuda()
            self.L1_loss.cuda()

    def build_dataloader(self):
        self.train_data_loader_A, self.train_data_loader_B = train_dataloader(input_size=self.input_size, batch_size=self.batch_size, dataset=self.dataset)
        self.test_data_loader_A, self.test_data_loader_B = test_dataloader(input_size=self.input_size, batch_size=self.batch_size, dataset=self.dataset)

    @staticmethod
    def save_image(filename, img):
        img = img.cpu().data[0].numpy()
        img *= 255.0
        img = img.clip(0, 255)
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
        img.save(filename)

    def save(self):
        if not os.path.exists('./' + self.dataset):
            os.mkdir('./' + self.dataset)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        torch.save(self.G_A.state_dict(), self.model_dir + 'generator_A_param.pkl')
        torch.save(self.G_B.state_dict(), self.model_dir + 'generator_B_param.pkl')
        torch.save(self.D_A.state_dict(), self.model_dir + 'discriminator_A_param.pkl')
        torch.save(self.D_B.state_dict(), self.model_dir + 'discriminator_B_param.pkl')

    def train(self):
        num_pool = 50
        fake_A_pool = misc.ImagePool(num_pool)
        fake_B_pool = misc.ImagePool(num_pool)

        for i, (real_A, real_B) in enumerate(zip(self.train_data_loader_A, self.train_data_loader_B)):
            self.G_A.train()
            self.G_B.train()

            # input image data
            real_A = Variable(real_A.cuda() if self.cuda else real_A)
            real_B = Variable(real_B.cuda()if self.cuda else real_B)

            # Train generator G
            # A -> B
            fake_B = self.G_A(real_A)
            D_B_fake_decision = self.D_B(fake_B)
            G_A_loss = self.MSE_loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda()))

            # forward cycle loss
            recon_A = self.G_B(fake_B)
            cycle_A_loss = self.L1_loss(recon_A, real_A) * self.lambdaA

            # B -> A
            fake_A = self.G_B(real_B)
            D_A_fake_decision = self.D_A(fake_A)
            G_B_loss = self.MSE_loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).cuda()))

            # backward cycle loss
            recon_B = self.G_A(fake_A)
            cycle_B_loss = self.L1_loss(recon_B, real_B) * self.lambdaB

            # Back propagation
            G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()

            # Train discriminator D_A
            D_A_real_decision = self.D_A(real_A)
            D_A_real_loss = self.MSE_loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).cuda()))
            fake_A = fake_A_pool.query(fake_A)
            D_A_fake_decision = self.D_A(fake_A)
            D_A_fake_loss = self.MSE_loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).cuda()))

            # Back propagation
            D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
            self.D_A_optimizer.zero_grad()
            D_A_loss.backward()
            self. D_A_optimizer.step()

            # Train discriminator D_B
            D_B_real_decision = self.D_B(real_B)
            D_B_real_loss = self.MSE_loss(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size()).cuda()))
            fake_B = fake_B_pool.query(fake_B)
            D_B_fake_decision = self.D_B(fake_B)
            D_B_fake_loss = self.MSE_loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).cuda()))

            # Back propagation
            D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
            self.D_B_optimizer.zero_grad()
            D_B_loss.backward()
            self.D_B_optimizer.step()

            misc.progress_bar(i, len(self.train_data_loader_A), 'D_A_loss: %.4f | D_B_loss: %.4f | G_A_loss: %.4f | G_B_loss: %.4f'
                              % (D_A_loss.data[0], D_B_loss.data[0], G_A_loss.data[0], G_B_loss.data[0]))

    def sample(self, model_path, pic_path, mode):
        transform = transforms.Compose([transforms.Resize(self.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        if mode == 'A':
            # setup model
            self.build_model()
            self.G_A.load_state_dict(torch.load(model_path))
            self.G_A.eval()

            out = self.G_A(Variable(transform(Image.open(pic_path)).unsqueeze(0).cuda(), volatile=True))

        else:
            # setup model
            self.build_model()
            self.G_B.load_state_dict(torch.load(model_path))
            self.G_B.eval()

            out = self.G_B(Variable(transform(Image.open(pic_path)).unsqueeze(0).cuda(), volatile=True))

        self.save_image('out.jpg', out)

    def run(self):
        self.build_model()
        print("model built")
        self.build_dataloader()
        print("dataloader prepared")

        for epoch in range(self.num_epochs):
            print("==> Epoch {}/{}".format(epoch + 1, self.num_epochs))
            if (epoch + 1) > self.decay_epoch:
                self.D_A_optimizer.param_groups[0]['lr'] -= self.lrD / (self.num_epochs - self.decay_epoch)
                self.D_B_optimizer.param_groups[0]['lr'] -= self.lrD / (self.num_epochs - self.decay_epoch)
                self.G_optimizer.param_groups[0]['lr'] -= self.lrG / (self.num_epochs - self.decay_epoch)

            self.train()
            self.save()
