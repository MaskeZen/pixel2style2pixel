
import os
import time
import logging
import numpy as np
import pickle

import sys, traceback

from numpy import asarray
from numpy import save

from PIL import Image
import torch
from torch.serialization import DEFAULT_PROTOCOL
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

    def format_stacktrace():
        parts = ["Traceback (most recent call last):\n"]
        parts.extend(traceback.format_stack(limit=25)[:-2])
        parts.extend(traceback.format_exception(*sys.exc_info())[1:])
        return "".join(parts)

    def process_image(input_image, net, experiment_args, latent_file_name='latent_file'):
        img_transforms = experiment_args['transform']
        input_image_transform = img_transforms(input_image)
        latent_mask = None

        with torch.no_grad():
            tic = time.time()
            result_image, result_latent = data_utils.run_on_batch(input_image_transform.unsqueeze(0), net, latent_mask)
            result_image = result_image[0]            
            torch.save(result_latent[0], "output/latent_test.pkl")
            try:
                torch_one = result_latent[0]
                numpy_array = torch_one.cpu().data.numpy()
                numpy_array = np.array([numpy_array])
                latent_file_output = "output/"+latent_file_name+".pkl"
                with open(latent_file_output, 'wb') as out_file:
                    pickle.dump(numpy_array, out_file)
                print(' ------- save pickle '+latent_file_output+' OK! ------- ')
            except:
                print('error on transform torch2npy')
                stacktrace = BootstrapPixel2Style.format_stacktrace()
                print(stacktrace)
            toc = time.time()
            print('Inference took {:.4f} seconds.'.format(toc - tic))

        output_image = tensor2im(result_image)

        res = np.array(output_image.resize((256, 256)))
        res_image = Image.fromarray(res)
        return res_image

    def get_latent_space(image, net, experiment_args):
        img_transforms = experiment_args['transform']
        input_image_transform = img_transforms(image)
        result_image, result_latent = data_utils.run_on_batch(input_image_transform.unsqueeze(0), net, None)
        result_image = result_image[0]
        print("result_latent.shape -> ")
        print(result_latent[0].shape)
        return result_latent[0]

    def process_image_interpolate(self, image1, image2, net, experiment_args):
        latent1 = self.get_latent_space(image1, net, experiment_args)
        latent2 = self.get_latent_space(image2, net, experiment_args)
        interpolated_latent_code = self.linear_interpolate(latent1, latent2, 0.5)
        interpolated_latent_code.shape    


    def linear_interpolate(code1, code2, alpha):
        return code1 * alpha + code2 * (1 - alpha)