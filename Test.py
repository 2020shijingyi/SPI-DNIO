import torch.nn as nn
import torch.optim as optim
from Util import *
import torch
import torch.backends.cudnn as cudnn
from Archs import Model
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import os
from tqdm import tqdm
from pytorch_msssim import ssim
cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# first pretrain model
model:Model = Model(res_num = 14,compression = 0.25).to(device)
#load weight type .pth
check = torch.load(r"",map_location='cpu')
sd = check['model_state_dict']

model.load_state_dict(sd,strict=True)
model.eval()
# the setting for the iterative optimization
weight_decay=0.1
mseloss = nn.MSELoss().to(device)
lr_o = 0.0001*0.5**2.2


num_epochs =400

# load the opt patterns   type .pt
opt_patterns = torch.load(r"").to(device)

result_save_path = '.\\output_noise\\'

### repeated experiment number#####
repeat_step = 10
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
for j in range(repeat_step):

    # early stop strategy
    # stopper, stop = EarlyStopping(patience=125), False
    model2: Model = Model(res_num = 14,compression = 0.25).to(device)
    optimizer = optim.Adam(model2.parameters(), lr=lr_o, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001 * 0.5 ** 6.6)
    result_save_path_step = result_save_path + str(j)+'\\'
    model2.to(device)
    psnr_best = 0.0
    ssim_best = 0.0

    if not os.path.exists(result_save_path_step):
        os.makedirs(result_save_path_step)
    record_csv = {'Loss': [], 'PSNR': [],'SSIM':[]}

    with tqdm(total=num_epochs, ncols=80) as t:
        for epoch in range(num_epochs):



            model2.train()
            #used to visualize performance
            epoch_losses = AverageMeter()
            epoch_psnr = AverageMeter()
            epoch_ssim = AverageMeter()

            if epoch ==0: #
                T = load_data(r"Data/Abdullah_Gul_0015.jpg").to(device)
                mea = measurement_encoding(T, patterns=opt_patterns).to(device)
                # if add noise
                # mea_noise = add_noise_to_mea(Normalized_max(mea), dSNR=10)
                # mea_noise = Normalized_max(mea_noise)
                # mea = mea_noise

                DGI_IN = DGI_reconstruction(mea,opt_patterns)
                DGI_IN =Normalized_std(DGI_IN)

                with torch.no_grad():
                   DGI_re = model.forward_(DGI_IN)
                mea = Normalized_std(mea)


            SPI_DNIO = model2.forward_(DGI_re) # forward

            SPI_DNIO = Normalized_std(SPI_DNIO)


            TV = total_variation(SPI_DNIO)

            mea_tilde = measurement_encoding(SPI_DNIO,opt_patterns)
            mea_tilde = Normalized_std(mea_tilde)


            loss = mseloss(mea_tilde, mea) + 1e-10 * TV

            loss.backward()
            epoch_losses.update(loss.item(), len(T))

            epoch_psnr.update(calc_psnr(SPI_DNIO,T),len(T))
            epoch_ssim.update(ssim(SPI_DNIO, T,data_range=1.0,size_average=True), len(T))

            optimizer.step()
            optimizer.zero_grad()

            # used to visualize results
            if epoch_ssim.avg > ssim_best:
                save_img(SPI_DNIO,name='SPI_DNIO_best',filepath=result_save_path_step)

            if epoch==0:
                save_img(T, name='T', filepath=result_save_path_step)
                save_img(DGI_IN,name='DGI_IN', filepath=result_save_path_step)
                DGI_re_s = Normalized_std(DGI_re)
                save_img(Normalized_std(DGI_re),name='DGI_first',filepath=result_save_path_step)

            if epoch %20 ==0:
                print('epoch_psnr{:.3f}'.format(epoch_psnr.avg))
                print('epoch_ssim{:.3f}'.format(epoch_ssim.avg))
            if epoch %100 ==0:
                save_img(SPI_DNIO, name='SPI_DNIO_%d'% (epoch), filepath=result_save_path_step)

            if epoch==399:
                save_img(SPI_DNIO, name='SPI_DNIO_%d'% (epoch), filepath=result_save_path_step)
                print('step:%d---learning rate:%f' % (epoch,optimizer.param_groups[0]['lr']))

            if epoch_ssim.avg > ssim_best:
                ssim_best = epoch_ssim.avg
            if epoch_psnr.avg > psnr_best:
                psnr_best = epoch_psnr.avg

            loss_value = epoch_losses.avg
            ssim_value = round(epoch_ssim.avg.item(), 4)
            psnr_value = round(epoch_psnr.avg.item(), 2)
            record_csv['Loss'].append(loss_value)
            record_csv['PSNR'].append(psnr_value)
            record_csv['SSIM'].append(ssim_value)

            scheduler.step()
            avg_loss = epoch_losses.avg
            #if early stop strategy
            # stop = stopper(epoch=epoch, fitness=epoch_ssim.avg)
            # if stop:
            #     save_img(SPI_DNIO, name='SPI_DNIO_%d' % (epoch), filepath=result_save_path_step)
            #     break  # must break all DDP ranks

            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))
            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update()

        print(f"SSIM-max:{ssim_best:.4f},PSNR-max:{psnr_best:.4f}")
        print(record_csv['SSIM'][-1])
        print(record_csv['PSNR'][-1])
        df = pd.DataFrame(record_csv)
        #performance metrics
        df.to_excel(os.path.join(result_save_path_step, 'Metrics.xlsx'), index=False)


