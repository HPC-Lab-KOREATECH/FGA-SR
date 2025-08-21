import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel
from io import BytesIO
import zipfile
import json

def load_img(img_path):
    window_size = 64 # Define windoes_size of D
    img_lr, img_hr = prepare_images(img_path, window_size=8)  # Change this image name

    tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]

    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

    img_lr, img_hr = prepare_images(img_path, window_size=8)  # Change this image name

    return img_lr, img_hr, cv2_lr, cv2_hr, tensor_lr

def cv2_to_byte_stream(cv_image):
    _, img_encoded = cv2.imencode('.png', cv_image)
    img_bytes = img_encoded.tobytes()
    img_io = BytesIO(img_bytes)
    img_io.seek(0)
    return img_io

def pil_to_byte_stream(pil_image):
    cv_image = pil_to_cv2(pil_image)
    return cv2_to_byte_stream(cv_image)

def get_position_image(img_hr, window_size, h, w):
    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (80, 176, 0), 2)
    return cv2_to_byte_stream(draw_img)

def get_diffusion_index(model_abs_normed_grad_numpy):
    gini_index = gini(model_abs_normed_grad_numpy)
    diffusion_index = (1 - gini_index) * 100
    return diffusion_index

def cal_lam(model, tensor_lr, img_lr, img_hr, h, w, window_size, data_range=1.):
    sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.5
    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy() * 2 - 1, model, attr_objective, gaus_blur_path_func, cuda=True) if data_range != 1. else Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
    if data_range != 1.:
        for i in range(len(result_numpy)):
            result_numpy[i] = result_numpy[i] / 2  + 0.5

    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    model_abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(model_abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(model_abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    images = [
        pil_to_byte_stream(saliency_image_abs),
        pil_to_byte_stream(blend_abs_and_input),
        pil_to_byte_stream(blend_kde_and_input),
        pil_to_byte_stream(Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.)))
    ]

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        zf.writestr('image_abs.png', images[0].getvalue())
        zf.writestr('blend_abs.png', images[1].getvalue())
        zf.writestr('blend_kde.png', images[2].getvalue())
        zf.writestr('tensor.png', images[3].getvalue())
        di = get_diffusion_index(model_abs_normed_grad_numpy)
        zf.writestr('data.json', json.dumps({"diffusionIndex": di}))

    memory_file.seek(0)
    return memory_file
