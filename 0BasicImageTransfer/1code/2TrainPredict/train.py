from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import time

print(torch.cuda.is_available())
# import our libraries
from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from dataset import get_training_set, get_test_set
# Training settings
'''
argparse is a module for CommandLineInterface,
its function, ArgumentParser, can pass user input into the program
example command:
	!python train.py --dataset facades --cuda
result：
	Namespace(batch_size=1, beta1=0.5, cuda=True, dataset='facades', 
	direction='b2a', epoch_count=1, input_nc=3, lamb=10, lr=0.0002, 
	lr_decay_iters=50, lr_policy='lambda', ndf=64, ngf=64, niter=100, 
	niter_decay=100, output_nc=3, seed=123, test_batch_size=1, threads=4)
'''

parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', type=str, default='FEA_p1', help='name of the data folder in <dataset> parent folder')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=80, help='# of iter to linearly increase learning rate from starting learning rate')
parser.add_argument('--niter_decay', type=int, default=80, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
# parser.add_argument('--lr_decay_iters', type=int, default=20, help='step: multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=1, help='weight on L1 term in objective')
parser.add_argument('--netG', type=str, default='unet_256', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
opt = parser.parse_args()

print(opt)

# check if cuda is available
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# open this allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# if NN structure is fixed during training, it is recommanded to open
cudnn.benchmark = True
# define random seed no.
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
# call function in data.py：to pass trainset address and transfer direction to DatasetFromFolder class
train_set = get_training_set(root_path + opt.dataset, opt.direction)
test_set = get_test_set(root_path + opt.dataset, opt.direction)
# pass trainset to loader
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)


device = torch.device("cuda:0" if opt.cuda else "cpu")
# initialize model
print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)
# initialize loss function
criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
# initialize optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# get learning rate updater
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

# initialize lists:
mseloss = ['test_mseloss']
lossD = ['train_dloss']
lossG = ['train_gloss']
# start training
start_time = time.time()
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    avg_loss_d = 0
    avg_loss_g = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        # get pair
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        # generate fake one
		fake_b = net_g(real_a)
        ######################
        # (1) Update D network
        ######################
        optimizer_d.zero_grad()
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        # see if we can predict this pair is fake
        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
		# see if we can predict this pair is real
        loss_d_real = criterionGAN(pred_real, True)
        # Combine them
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        # use backward function to autocalculate dLoss/dW
        loss_d.backward()
        # update parameter's value from optimizer
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################
        optimizer_g.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        # see if we can generate the real image
        loss_g_gan = criterionGAN(pred_fake, True)
        # compare it against the original one for l1 loss
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        optimizer_g.step()
        # print loss on this image sample
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
        # sum loss over all samples
        avg_loss_d += loss_d.item()
        avg_loss_g += loss_g.item()
        
    
    lossD += [avg_loss_d / len(testing_data_loader)]
    lossG += [avg_loss_g / len(testing_data_loader)]
    print("===> Avg. lossD: {:.4f}".format(avg_loss_d / len(testing_data_loader)) )
    print("===> Avg. lossG: {:.4f}".format(avg_loss_g / len(testing_data_loader)) )
    
    # update learning rate for every epoch
    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test model performance via Peak Signal to Noise Ratio
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)
        # a to b
        prediction = net_g(input)
        # compare
        mse = criterionMSE(prediction, target)
        psnr = - 10 * log10(mse.item())	#because the image is normalized, so max pixel value is 1, so 20*log10(maxI)=0
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    psnr += [avg_psnr]

    # save model and psnr in checkpoints for every # epochs
    if epoch % 5 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
           
        with open("checkpoint/psnr.txt", 'w') as f:
            for item in zip(lossD, lossG, psnr):
                f.write("{}\n".format(item))
    
print('Training Time Taken: {} hours'.format((time.time() - start_time)/60/60) )
