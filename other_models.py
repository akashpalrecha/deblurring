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
        self.hparams['n_colors']     = args['n_colors']
        self.hparams['res_scale']    = args['res_scale']
        
        self.model = EDSR(args['n_resblocks'], args['n_feats'], args['n_colors'], args['res_scale'])
        
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = DeblurModelBase.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_resblocks', type=int, default=32)
        parser.add_argument('--n_feats', type=int, default=256)
        parser.add_argument('--n_colors', type=int, default=3)
        parser.add_argument('--res_scale', type=float, default=0.1)
        parser.set_defaults(lr=0.0001)
        parser.set_defaults(max_epochs=300)
        parser.set_defaults(lr_decay_every_n_epochs=60)
        parser.set_defaults(lr_decay_factor=0.5)
        return parser