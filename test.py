from argparse import ArgumentParser
import pytorch_lightning as pl
from models import *
from typing import Callable
from data import *
import os
from pathlib import Path
import torchvision
import mimetypes
from PIL import Image
from pdb import set_trace
try:
    get_ipython().__class__.__name__
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

models_dict = {"sample_model": SampleModel,
               "simple_cnn": SimpleCNNModel}

datasets_dict = {'levin':'sample_levin_dataset',
                 'GOPRO': None,
                 'lai': None}

stats_dict    = {'levin': None,
                 'GOPRO': None,
                 'lai': None}

def image_filter(path:str):
    path = str(path)
    ftype = mimetypes.guess_type(path)[0]
    if type(ftype) is str:
        return 'image' in ftype
    elif ftype is None:
        return False
    else:
        return False

toTensor = torchvision.transforms.ToTensor()

def read_image(path:Path, use_gpu=False):
    im = Image.open(str(path))
    if use_gpu:
        return toTensor(im).cuda()
    else:
        return toTensor(im)

if __name__ == '__main__':
    """
    Takes in input folder for predicting on images
    Takes in output folder for predictions
    Takes in experiment directory
    """
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing images to predict on")
    parser.add_argument("--output_folder",type=str, default="0",
                        help="Folder to output predictions to")
    parser.add_argument("--experiment_folder", type=str, required=True,
                        help="Folder containing the saved model to use")
    parser.add_argument("--use_gpu", action='store_true')
    
    args = vars(parser.parse_args())
    
    in_folder   = Path(args['input_folder'])
    saved_model = Path(args['experiment_folder'])
    
    out_folder  = args['output_folder']
    if out_folder == "0": # Get default out folder
        out_folder = in_folder.name + "_test_output"
        out_folder = in_folder.parent/out_folder
    out_folder = Path(out_folder)
    os.makedirs(str(out_folder), exist_ok=True)
    
    print(f"Input: {in_folder}")
    print(f"Output: {out_folder}")
    print(f"Loading model from: {saved_model}")
    
    model_info  = torch.load(str(saved_model/'model_info.pt'))
    model_name  = model_info['model_name']
    data_module = model_info['data_module']
    dargs       = model_info['dargs']
    
    model_fn    = models_dict[model_info['model_name']]
    model:DeblurModelBase = model_fn(data_module, args=dargs)
    
    checkpoint = next((saved_model/'checkpoints').iterdir()) # Only best model is saved in checkpoint
    checkpoint = torch.load(str(checkpoint))
    model.load_state_dict(checkpoint['state_dict'])
    if args['use_gpu']:
        model = model.cuda()
    print("Successfully loaded model")
    
    image_list = list(filter(image_filter, in_folder.iterdir()))
    
    for impath in tqdm(image_list):
        try:
            im = read_image(impath, args['use_gpu'])
            out = model.predict_image(im, normalized=False)
            out = (out.detach().cpu().numpy() * 255).astype(np.uint8)
            # set_trace()
            if out.shape[2] == 1:
                out = Image.fromarray(out[:, :, 0], 'L')
            else:
                out = Image.fromarray(out)
            out.save(str(out_folder/impath.name))
        except:
            print(f"Error when working with : {str(impath)}")
            raise Exception("testing aborted")
        
    print("Done!")
        