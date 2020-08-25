from argparse import ArgumentParser
import pytorch_lightning as pl
from models import *
from typing import Callable
from data import *
from pytorch_lightning.callbacks import ModelCheckpoint
import os

models_dict = {"sample_model": SampleModel,
               "simple_cnn": SimpleCNNModel}

datasets_dict = {'levin':'sample_levin_dataset',
                 'GOPRO': None,
                 'lai': None}
stats_dict    = {'levin': None,
                 'GOPRO': None,
                 'lai': None}

def main(args):
    dargs = vars(args)
    
    dataset_info = datasets_dict.get(dargs['dataset'], dargs['dataset'])
    other_args = {'batch_size': dargs['batch_size'],
                  'val_pct': dargs['val_pct'],
                  'size': dargs['size'],
                  'stats': stats_dict.get(dargs['dataset'], None),
                  'valid_batch_size': dargs['valid_batch_size']}
    name = dargs['dataset']
    if type(dataset_info) == str:
        data_module = DeblurDataModule(dataset_info, name=name, **other_args)
    elif type(dataset_info) == list:
        data_module = DeblurDataModule(*dataset_info, name=name, **other_args)
    elif type(dataset_info) == type(lambda x: x):
        train_files, test_files = dataset_info()
        data_module = DeblurDataModule(train_files=train_files, test_files=test_files, 
                                       name=name, **other_args)
    else:
        raise Exception(f"Could not load dataset with argument: {dargs['dataset']}")

    print(f"Loaded dataset with {len(data_module.train_files)} training images and \
          {len(data_module.valid_files)} validation/testing images.")

    model_fn:Callable = models_dict[dargs['model_name']]
    model:DeblurModelBase = model_fn(data_module, args=dargs)
    
    # Preparing logger and logging hyperparameters and one batch of data
    logger = pl.loggers.TensorBoardLogger(save_dir='experiments', 
                                          name=model.hparams['exp_name'])
    logger.log_hyperparams(params=model.hparams,
                           metrics={"PSNR":0.0, "SSIM": 0.0, "val_loss":0.0,
                                    "epoch_test_PSNR":0.0, "epoch_test_SSIM":0.0})
    figure = data_module.show_batch(4)
    plt.savefig(os.path.join(logger.log_dir, 'one_batch.png'))
    logger.experiment.add_figure(tag=f"{4} batches of training data", figure=figure, global_step=0)
    
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(logger.log_dir, 'checkpoints', 'best-{epoch}-{val_loss:.2f}.ckpt'))
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, checkpoint_callback=checkpoint_callback)
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='simple_cnn')
    parser.add_argument('--dataset', type=str, default='levin')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--valid_batch_size', type=int, default=0)
    parser.add_argument('--val_pct', type=float, default=0.2)
    parser.add_argument('--size', nargs="+", type=int, default=0)
    parser.add_argument('--tag', type=str, default="")
    parser = pl.Trainer.add_argparse_args(parser)

    temp_args, _ = parser.parse_known_args()

    model_fn = models_dict.get(temp_args.model_name, False)
    if model_fn:
        parser = model_fn.add_model_specific_args(parser)
    else:
        raise Exception(f"No model found with model_name: {temp_args.model_name}")

    args = parser.parse_args()

    main(args)
    # args = parser.parse_args()total_imagestotal_imagestotal_images