
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

class BootstrapPixel2Style:

    logging.basicConfig(filename="pixel2style.log")

    #----------------------------------
    def load_model(task_type):
        print("loading model for task " + task_type)

        # Set the arguments from the task type.
        experiment_args = EXPERIMENT_DATA_ARGS[task_type]

        if os.path.exists(experiment_args['model_path']):
            check_file(experiment_args['model_path'])
        else:
            message = 'model is not in the folder: ' + experiment_args['model_path']
            print(message)
            logging.warning(message)
            download_pretrained_model(MODEL_PATHS_DOWNLOAD[task_type])

        net, opts = load_pretrained_model(experiment_args['model_path'])
        data_utils.download_dlib()

        return net, opts
    #-------------------------------
    def get_input_image(image_path, image_align = False):
        input_image = None
        if image_align:
            try:
                input_image = data_utils.align_image(image_path)
            except:
                message = "ERROR: when tried to align the image " + image_path
                print(message)
                logging.error(message)
        else:
            input_image = Image.open(image_path)
        return input_image
    #-------------------------------
    def process_image(input_image, net, experiment_args):
        img_transforms = experiment_args['transform']
        input_image_transform = img_transforms(input_image)

        # if task_type in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
        #     latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        # else:
        #     latent_mask = None
        latent_mask = None

        with torch.no_grad():
            tic = time.time()
            result_image = data_utils.run_on_batch(input_image_transform.unsqueeze(0), net, latent_mask)[0]
            toc = time.time()
            print('Inference took {:.4f} seconds.'.format(toc - tic))

        output_image = tensor2im(result_image)

        res = np.array(output_image.resize((256, 256)))
        res_image = Image.fromarray(res)
        return res_image
