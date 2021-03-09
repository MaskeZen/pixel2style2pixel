from argparse import Namespace
import os
import pprint
import torch

from models.psp import pSp

CODE_DIR = os.getcwd()

def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
    current_directory = os.getcwd()
    save_path = os.path.join(os.path.dirname(current_directory), CODE_DIR, "pretrained_models")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url

# simple check to know if the model is OK.
def check_file(model_path):
    if os.path.getsize(model_path) < 1000000:
        raise ValueError("Pretrained model was unable to be downlaoded correctly!")

def download_pretrained_model(download_keys):
    download_command = get_download_model_command(file_id=download_keys["id"], file_name=download_keys["name"])
    os.system(download_command)
    print('Model successfully downloaded!')

def load_pretrained_model(model_path):
    check_file(model_path)
    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']
    pprint.pprint(opts)

    # update the training options
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    return net, opts