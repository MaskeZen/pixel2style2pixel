
import os
import time
import logging
import numpy as np
from PIL import Image
import torch
from utils.common import tensor2im
from options.pixel2style_options import Pixel2StyleOptions
from configs.pixel2style_config import MODEL_PATHS_DOWNLOAD, EXPERIMENT_DATA_ARGS
from models.model_download import check_file,download_pretrained_model, load_pretrained_model
from utils import data_utils

CODE_DIR = os.getcwd()
logging.basicConfig(filename="pixel2style.log")

# Initialize the aplication options
opts = Pixel2StyleOptions().parse()
task_type = opts.task
image_align = opts.align
input_path = opts.imagen
#----------------------------------

if input_path == "" or input_path == None or (not os.path.isdir(input_path) and not data_utils.is_image_file(input_path)):
    raise ValueError("The file pased as argument is not a valid image.")

print("starting task: " + task_type)

# Set the arguments from the task type.
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[task_type]

if os.path.exists(EXPERIMENT_ARGS['model_path']):
    check_file(EXPERIMENT_ARGS['model_path'])
else:
    message = 'model is not in the folder: ' + EXPERIMENT_ARGS['model_path']
    print(message)
    logging.warning(message)
    download_pretrained_model(MODEL_PATHS_DOWNLOAD[task_type])

net, opts = load_pretrained_model(EXPERIMENT_ARGS['model_path'])
data_utils.download_dlib()

images_path = []
if os.path.isdir(input_path):
    images_path = data_utils.make_dataset(input_path)
else:
    images_path.append(input_path)

inputs_images = {}
for image_path in images_path:
    if image_align:
        try:
            input_image = data_utils.align_image(image_path)
        except:
            message = "ERROR: when tried to align the image " + image_path
            print(message)
            logging.error(message)
            continue
    else:
        input_image = Image.open(image_path)

    img_transforms = EXPERIMENT_ARGS['transform']
    inputs_images[image_path] = img_transforms(input_image)

    if task_type in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
        latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    else:
        latent_mask = None

    with torch.no_grad():
        tic = time.time()
        result_image = data_utils.run_on_batch(inputs_images[image_path].unsqueeze(0), net, latent_mask)[0]
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    output_image = tensor2im(result_image)

    res = np.array(output_image.resize((256, 256)))
    res_image = Image.fromarray(res)
    if not os.path.exists('output'):
        os.mkdir('output')
        print("se creo el directorio output.")
    
    base_path, image_file_name = os.path.split(image_path)
    res_image.save("output/"+task_type+"_"+image_file_name+".jpg")

print("    - -- ---= TASK COMPLETE =--- -- - ")
