import torchgeometry as tgm
import torch


loss_func_ssim = tgm.losses.SSIM(5, reduction='none')

outputs = torch.rand(1, 3, 256, 256)
imgs = torch.rand(1, 3, 256, 256)

loss_SSIM = loss_func_ssim(outputs, outputs)
loss_pixel = torch.mean(loss_SSIM)
print('A')
