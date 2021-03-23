# Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation
  <a href="https://arxiv.org/abs/2008.00951"><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/eladrich/pixel2style2pixel/blob/master/notebooks/inference_playground.ipynb)

> We present a generic image-to-image translation framework, Pixel2Style2Pixel (pSp). Our pSp framework is based on a novel encoder network that directly generates a series of style vectors which are fed into a pretrained StyleGAN generator, forming the extended W+ latent space. We first show that our encoder can directly embed real images into W+, with no additional optimization. We further introduce a dedicated identity loss which is shown to achieve improved performance in the reconstruction of an input image. We demonstrate pSp to be a simple architecture that, by leveraging a well-trained, fixed generator network, can be easily applied on a wide-range of image-to-image translation tasks. Solving these tasks through the style representation results in a global approach that does not rely on a local pixel-to-pixel correspondence and further supports multi-modal synthesis via the resampling of styles. Notably, we demonstrate that pSp can be trained to align a face image to a frontal pose without any labeled data, generate multi-modal results for ambiguous tasks such as conditional face generation from segmentation maps, and construct high-resolution images from corresponding low-resolution images.

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>
</p>

## Description   
Official Implementation of our pSp paper for both training and evaluation. The pSp method extends the StyleGAN model to 
allow solving different image-to-image translation problems using its encoder.

## Dependencies

We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/psp_env.yaml`.

1. Create environment:

`conda env create -f environment/psp_env.yaml`

2. Activate:

`conda activate psp_env`

## CMD

```cmd
usage: cmd.py [-h] [-i IMAGEN] [-a]
              [-t {task type}]

optional arguments:
  -h, --help            *show this help message and exit*
  -i, --imagen IMAGEN   *Path to the image*
  -a, --align           *If is necesary align the images.*
  -t, --task TASK-TYPE  *Type of task to run*
    {ffhq_encode,ffhq_frontalize,celebs_sketch_to_face,celebs_seg_to_face,celebs_super_resolution,toonify}
```

## Server

To use the server install the Flask package is required.
`pip install flask`

Then setup the FLASK_APP environment variable.
`export FLASK_APP=server.py`

And run the server (add --host=0.0.0.0 to allow access from hosts in the same network).
`flask run --host=0.0.0.0`

### Testing the API

With curl is posible send a image and save the response with the response header `Content-Disposition`:
Request:
`curl -v -OJ -X POST -F 'imagen=@./Pedro_Dalton-00002.jpg' http://myflaskserver:5000/frontalize`
Response header:
`Content-Disposition: attachment; filename=Pedro_Dalton-00002.`

## TODO

- [ ] Delete the image after send the response, or send directly from memory. (https://stackoverflow.com/questions/40853201/remove-file-after-flask-serves-it)
