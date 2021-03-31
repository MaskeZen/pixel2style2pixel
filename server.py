import os
import logging

from flask import Flask
from flask.globals import request
from flask.helpers import send_file
from werkzeug.utils import secure_filename
from bootstrap import BootstrapPixel2Style
from configs.pixel2style_config import EXPERIMENT_DATA_ARGS

app = Flask(__name__)

UPLOAD_FOLDER = './inputs'
if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

## ------------------------------------------------------------------------------------------------

CODE_DIR = os.getcwd()
logging.basicConfig(filename="pixel2style.log")

# Initialize the aplication options
task_type = 'ffhq_frontalize'
image_align = True
input_path = ''
#----------------------------------

net_ffhq_frontalize, opts = BootstrapPixel2Style.load_model(task_type)

net_ffhq_encode, opts = BootstrapPixel2Style.load_model('ffhq_encode')

## ------------------------------------------------------------------------------------------------

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/frontalize', methods=['GET', 'POST'])
def frontalize():
    if request.method == 'GET':
        return 'mostrar ayuda'
    else:
        imagen = request.files['imagen']
        if imagen != None:
            filename = secure_filename(imagen.filename)
            tmp_file_path = os.path.join(UPLOAD_FOLDER, filename)
            imagen.save(os.path.join(UPLOAD_FOLDER, filename))

            logging.info('frontalizando imagen...' + filename)

            input_image = BootstrapPixel2Style.get_input_image(tmp_file_path, image_align)
            if input_image == None:
                return 'Error al frontalizar la imagen'
            res_image = BootstrapPixel2Style.process_image(input_image, net_ffhq_frontalize, EXPERIMENT_DATA_ARGS[task_type])

            if not os.path.exists('output'):
                os.mkdir('output')
                print("se creo el directorio output.")
            
            base_path, image_file_name = os.path.split(tmp_file_path)
            output_file = "output/flask_"+task_type+"_"+image_file_name+".jpg"
            res_image.save(output_file)
            res_image.tobytes()
            return send_file(filename_or_fp=output_file,attachment_filename=image_file_name+"_"+task_type+".jpg",as_attachment=True)
        else:
            return 'no se envio ninguna imagen'

@app.route('/encode', methods=['GET', 'POST'])
def encode():
    task_type = 'ffhq_encode'
    if request.method == 'GET':
        return 'mostrar ayuda'
    else:
        imagen = request.files['imagen']
        if imagen != None:
            input_image, filename = load_image(imagen)
            
            res_image = BootstrapPixel2Style.process_image(input_image, net_ffhq_encode, EXPERIMENT_DATA_ARGS[task_type])

            if not os.path.exists('output'):
                os.mkdir('output')
                print("se creo el directorio output.")
            
            tmp_file_path = os.path.join(UPLOAD_FOLDER, filename)
            base_path, image_file_name = os.path.split(tmp_file_path)
            output_file = "output/flask_"+task_type+"_"+image_file_name+".jpg"
            res_image.save(output_file)
            res_image.tobytes()
            return send_file(filename_or_fp=output_file,attachment_filename=image_file_name+"_"+task_type+".jpg",as_attachment=True)
        else:
            return 'no se envio ninguna imagen'

def load_image(imagen):
    if imagen != None:
        filename = secure_filename(imagen.filename)
        tmp_file_path = os.path.join(UPLOAD_FOLDER, filename)
        imagen.save(os.path.join(UPLOAD_FOLDER, filename))

        logging.info('frontalizando imagen...' + filename)

        input_image = BootstrapPixel2Style.get_input_image(tmp_file_path, image_align)
        # if input_image == None:
        #     return 'Error al frontalizar la imagen'
        return input_image, filename
    
