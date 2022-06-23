import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
from model.utils import DataLoader
from utils import *
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="anomaly detection using aemem")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size for training')
parser.add_argument('--test_batch_size', type=int,
                    default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60,
                    help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.1,
                    help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1,
                    help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256,
                    help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate')
parser.add_argument('--method', type=str, default='pred',
                    help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5,
                    help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512,
                    help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512,
                    help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10,
                    help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1,
                    help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2',
                    help='type of dataset: ped1, ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str,
                    default='./dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log',
                    help='directory of log')

start_time = datetime.now()
print("Start time:", start_time.strftime("%d/%m/%Y %H:%M:%S"))

args = parser.parse_args()

print("Dataset: ", args.dataset_type)
print("Method: ", args.method)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

# make sure to use cudnn for computational performance
torch.backends.cudnn.enabled = True

train_folder = args.dataset_path+"/"+args.dataset_type+"/training/frames"
test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# Loading dataset
print('Loading dataset...')
train_dataset = DataLoader(train_folder, transforms.Compose([
    transforms.ToTensor(),
]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_dataset = DataLoader(test_folder, transforms.Compose([
    transforms.ToTensor(),
]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)
print('Loading dataset is finished')


# Model setting
print('Model setting...')
assert args.method == 'pred' or args.method == 'recons', 'Wrong task name'
if args.method == 'pred':
    from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
else:
    from model.Reconstruction import *
    model = convAE(args.c, memory_size=args.msize,
                   feature_dim=args.fdim, key_dim=args.mdim)
params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
model.cuda()
print('Setting up model is finished')

print('Start training and logging into file')
# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'w')
sys.stdout = f

loss_func_mse = nn.MSELoss(reduction='none')

# Training
print('Start training...')
m_items = F.normalize(torch.rand((args.msize, args.mdim),
                      dtype=torch.float), dim=1).cuda()  # Initialize the memory items

for epoch in range(args.epochs):
    labels_list = []
    model.train()

    start = time.time()
    for j, (imgs) in enumerate(train_batch):
        # plt.imshow(imgs[0].permute(1, 2, 0))
        # plt.show()
        imgs = Variable(imgs).cuda()

        if args.method == 'pred':
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(
                imgs[:, 0:12], m_items, True)

        else:
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(
                imgs, m_items, True)

        optimizer.zero_grad()
        if args.method == 'pred':
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 12:]))
        else:
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs))

        loss = loss_pixel + args.loss_compact * compactness_loss + \
            args.loss_separate * separateness_loss
        loss.backward(retain_graph=True)
        optimizer.step()

    scheduler.step()

    print('----------------------------------------')
    print('Epoch:', epoch+1)
    if args.method == 'pred':
        print('Loss: Prediction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(
            loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    else:
        print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(
            loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')

print('Training is finished')
# Save the model and the memory items
prefix_output_name = args.dataset_type
if args.method == 'pred':
    prefix_output_name = prefix_output_name + '_prediction_'
else:
    prefix_output_name = prefix_output_name + '_reconstruction_'
torch.save(model, os.path.join(log_dir, prefix_output_name + 'model.pth'))
torch.save(m_items, os.path.join(log_dir, prefix_output_name + 'keys.pt'))

sys.stdout = orig_stdout
f.close()

print('Training is finished')
end_time = datetime.now()
time_range = end_time-start_time
print('Training is taken: ', time_range)
