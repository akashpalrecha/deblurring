from models import *
from extras.edsr import EDSR
import torch.nn as nn

class EDSR_Model(DeblurModelBase):
    def __init__(self, data_module:DeblurDataModule=None, args:dict={}):
        super(EDSR_Model, self).__init__(data_module, args)
        
        self.loss_func = nn.L1Loss()
        
        self.model_name = f"EDSR"
        self.hparams['model_name']   = self.model_name
        self.hparams['exp_name']     = self.model_name + '_' + (data_module.name if data_module else "") \
                                       + '_' + args.get('tag', '')
        self.hparams['n_resblocks']  = args['n_resblocks']
        self.hparams['n_feats']      = args['n_feats']
        self.hparams['n_colors']      = args['n_colors']
        self.hparams['res_scale']      = args['res_scale']
        
        self.model = EDSR(args['n_resblocks'], args['n_feats'], args['n_colors'], args['res_scale'])
        
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_resblocks', type=int, default=32)
        parser.add_argument('--n_feats', type=int, default=256)
        parser.add_argument('--n_colors', type=int, default=3)
        parser.add_argument('--res_scale', type=float, default=0.1)
        return parser