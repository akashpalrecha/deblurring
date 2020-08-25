import torch
import pytorch_lightning as pl
from data import *
from pytorch_lightning.metrics.functional import psnr
from pytorch_lightning.metrics.regression import SSIM
import time
from argparse import ArgumentParser
import torch.nn as nn
import os
from pdb import set_trace

class DeblurModelBase(pl.LightningModule):
    def __init__(self, data_module:DeblurDataModule, lr:float=0.001, args:dict={}):
        super(DeblurModelBase, self).__init__()
        self.data_module  = data_module
        self.transforms   = self.data_module.transforms
        self.v_transforms = self.data_module.valid_transforms
        self.loss_func    = torch.nn.MSELoss()
        self.metrics      = [(psnr, 'PSNR'), (SSIM(kernel_size=(5,5)), 'SSIM')]
        self.lr           = lr
        self.model_name   = "Model"
        self.test_out     = None # Folder for testing results
        self.hparams      = {}
        self.hparams['model_name']         = self.model_name
        self.hparams['dataset_name']       = data_module.name
        self.hparams['dataset_train_size'] = len(data_module.train_files)
        self.hparams['dataset_valid_size'] = len(data_module.valid_files)
        self.hparams['exp_name']           = self.model_name + '_' + data_module.name + '_' + args.get('tag', '')
        self.hparams['init_lr']            = self.lr
        
    def training_step(self, batch, batch_idx):
        x, y   = augment_image_pair(batch, self.transforms)
        out    = self(x)
        loss   = self.loss_func(out, y)
        result = pl.TrainResult(loss)

        result.log_dict({'train_loss':loss}, on_step=True, on_epoch=True,
                        prog_bar=True, logger=True, sync_dist=True)
        self.logger.log_metrics({f'train_loss/epoch_{self.current_epoch}': loss}, step=self.global_step)

        return result
    
    def validation_step(self, batch, batch_idx):
        # Only apply tfms for valid data if explicitly specified
        x, y = augment_image_pair(batch, self.v_transforms)
        out  = self(x)
        loss = self.loss_func(out, y)
        result = pl.EvalResult(checkpoint_on=loss)
        log_dict = {'val_loss': loss}
        
        # Calculate metrics like PSNR, SSIM, etc.
        if self.metrics:
            out = self.data_module.denormalize_func(out).clamp(0.0, 1.0)
            y   = self.data_module.denormalize_func(y).clamp(0.0, 1.0)
            for metric in self.metrics:
                log_dict[metric[1]] = metric[0](out, y)

        result.log_dict(log_dict, on_step=False, on_epoch=True, 
                        prog_bar=True, logger=True, sync_dist=True)

        return result
    
    def on_test_epoch_start(self):
        if self.test_out is None:
            self.test_out = os.path.join(self.logger.log_dir, 'test_results')
            os.makedirs(self.test_out, exist_ok=True)
    
    def test_step(self, batch, batch_idx):
        # Only apply tfms for valid data if explicitly specified
        x, y = augment_image_pair(batch, self.v_transforms)
        out  = self(x)
        loss = self.loss_func(out, y)
        result = pl.EvalResult(loss)
        log_dict = {'test_loss': loss}
        
        # Calculate metrics like PSNR, SSIM, etc.
        if self.metrics:
            pred = self.data_module.denormalize_func(out).clamp(0.0, 1.0)
            y   = self.data_module.denormalize_func(y).clamp(0.0, 1.0)
            for metric in self.metrics:
                log_dict[f"test_{metric[1]}"] = metric[0](pred, y)

        result.log_dict(log_dict, on_step=True, on_epoch=True, 
                        prog_bar=True, logger=True, sync_dist=True)
        
        # Saving predictions as images
        offset = batch_idx * self.data_module.valid_batch_size
        # As test/valid dataloaders are not shuffled
        fnames = self.data_module.valid_files[offset : offset + out.shape[0]]
        fnames = [Path(f[0]).name for f in fnames]
        images = self.data_module.decode(out)[0]
        # set_trace()
        
        for fname, image in zip(fnames, images):
            image = (image * 255).clip(0, 255).astype(np.uint8)
            if image.shape[2] == 1:
                Image.fromarray(image[:,:,0], 'L').save(os.path.join(self.test_out, fname))
            else:
                Image.fromarray(image).save(os.path.join(self.test_out, fname))
            # plt.imsave(os.path.join(self.test_out, fname), image, cmap='gray')

        return result
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parent_parser
    
    def forward(self, x):
        raise NotImplementedError


class SampleModel(DeblurModelBase):
    def __init__(self, data_module:DeblurDataModule, lr:float=0.001, args:dict={}):
        super(SampleModel, self).__init__(data_module, lr, args)
        self.a = torch.nn.Parameter(data=torch.tensor(1.0))
        self.model_name = "simple_scaler"
        self.hparams['model_name']   = self.model_name
        self.hparams['exp_name']     = self.model_name + '_' + data_module.name + '_' + args.get('tag', '')
    
    def forward(self, x):
        return self.a * x
    
    
class EDSRResBlock(nn.Module):
    def __init__(self, channels, ks=3, stride=1, padding=1):
        super(EDSRResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, 
                          kernel_size=ks, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, 
                          kernel_size=ks, stride=stride, padding=padding)
        
    def forward(self, x):
        return self.conv2(self.relu1(self.conv1(x))) + x
    
class SimpleCNNModel(DeblurModelBase):
    def __init__(self, data_module:DeblurDataModule, lr:float=0.001, args:dict={}):
        super(SimpleCNNModel, self).__init__(data_module, lr, args)
        in_ch = args['in_channels']
        num_edsr_blocks = args['num_edsr_blocks']
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.edsr_blocks = nn.Sequential()
        for i in range(num_edsr_blocks):
            self.edsr_blocks.add_module(f'edsr_{i+1}', EDSRResBlock(16))
        
        self.conv_final = nn.Conv2d(16, in_ch, kernel_size=3, stride=1, padding=1)
        
        self.model_name = f"simple_cnn"
        self.hparams['model_name']   = self.model_name
        self.hparams['exp_name']     = self.model_name + '_' + data_module.name + '_' + args.get('tag', '')
        self.hparams['in_channels']  = in_ch
        self.hparams['edsr_blocks']  = num_edsr_blocks
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.edsr_blocks(x)
        return self.conv_final(x)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--num_edsr_blocks', type=int, default=1)
        return parser