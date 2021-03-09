
import os

from argparse import Namespace
import os
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp

from options.pixel2style_options import Pixel2StyleOptions
from configs.pixel2style_config import MODEL_PATHS_DOWNLOAD, EXPERIMENT_DATA_ARGS
from models.model_download import check_file,download_pretrained_model, load_pretrained_model
from utils import data_utils

CODE_DIR = os.getcwd()

# Inicializa las opciones de la aplicación
opts = Pixel2StyleOptions().parse()

task_type = opts.task
image_align = opts.align
image_path = opts.imagen

# if os.path.isdir(dir):
if image_path == "" or image_path == None or not data_utils.is_image_file(image_path):
    raise ValueError("The file pased as argument is not a valid image.")


print("starting task: " + task_type)

# Set the arguments from the task type.
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[task_type]

if os.path.exists(EXPERIMENT_ARGS['model_path']):
    check_file(EXPERIMENT_ARGS['model_path'])
else:
    print('model is not in the folder: ' + EXPERIMENT_ARGS['model_path'])
    download_pretrained_model(MODEL_PATHS_DOWNLOAD[task_type])

net, opts = load_pretrained_model(EXPERIMENT_ARGS['model_path'])

data_utils.download_dlib()

if image_align:
    input_image = data_utils.align_image(image_path)
else:
    input_image = Image.open(image_path)
# if task_type not in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
#   input_image = data_utils.align_image(image_path)
# else:
#   input_image = image_path

img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image)

if task_type in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
    latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
else:
    latent_mask = None

def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch

with torch.no_grad():
    tic = time.time()
    result_image = run_on_batch(transformed_image.unsqueeze(0), net, latent_mask)[0]
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))

input_vis_image = log_input_image(transformed_image, opts)
output_image = tensor2im(result_image)

if task_type == "celebs_super_resolution":
    res = np.concatenate([np.array(input_image.resize((256, 256))),
                          np.array(input_vis_image.resize((256, 256))),
                          np.array(output_image.resize((256, 256)))], axis=1)
else:
    res = np.concatenate([np.array(input_vis_image.resize((256, 256))),
                          np.array(output_image.resize((256, 256)))], axis=1)

res_image = Image.fromarray(res)
if not os.path.exists('output'):
    os.mkdir('output')
    print("se creo el directorio output.")
res_image.save("output/res_image.jpg")