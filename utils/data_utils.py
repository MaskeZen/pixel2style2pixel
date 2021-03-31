"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import torch
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def download_dlib():
    dblibFile = "shape_predictor_68_face_landmarks.dat.bz2"
    if os.path.exists(dblibFile):
        print("ya se descargo dlib")
    else:
        os.system("wget http://dlib.net/files/" + dblibFile)
        os.system("bzip2 -dk " + dblibFile)

def align_image(image_path):
  import dlib
  from scripts.align_all_parallel import align_face
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image

def run_on_batch(inputs, net, latent_mask=None):
    # if latent_mask is None:
    result_batch, result_latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    # else:
    #     result_batch = []
    #     for image_idx, input_image in enumerate(inputs):
    #         # get latent vector to inject into our input image
    #         vec_to_inject = np.random.randn(1, 512).astype('float32')
    #         _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
    #                                   input_code=True,
    #                                   return_latents=True)
    #         # get output image with injected style vector
    #         res = net(input_image.unsqueeze(0).to("cuda").float(),
    #                   latent_mask=latent_mask,
    #                   inject_latent=latent_to_inject)
    #         result_batch.append(res)
    #     result_batch = torch.cat(result_batch, dim=0)
    return result_batch, result_latents
