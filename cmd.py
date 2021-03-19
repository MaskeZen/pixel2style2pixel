
import os
import logging
from options.pixel2style_options import Pixel2StyleOptions
from configs.pixel2style_config import EXPERIMENT_DATA_ARGS
from utils import data_utils
from bootstrap import BootstrapPixel2Style

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

net, opts = BootstrapPixel2Style.load_model(task_type)

images_path = []
if os.path.isdir(input_path):
    images_path = data_utils.make_dataset(input_path)
else:
    images_path.append(input_path)

inputs_images = {}
for image_path in images_path:
    input_image = BootstrapPixel2Style.get_input_image(image_path, image_align)
    if input_image == None:
        continue
    res_image = BootstrapPixel2Style.process_image(input_image, net, EXPERIMENT_DATA_ARGS[task_type])

    if not os.path.exists('output'):
        os.mkdir('output')
        print("se creo el directorio output.")
    
    base_path, image_file_name = os.path.split(image_path)
    res_image.save("output/"+task_type+"_"+image_file_name+".jpg")

print("    - -- ---= TASK COMPLETE =--- -- - ")
