import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
from math import sqrt, floor, ceil


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.
    After clamping to [min, max], values will be normalized to [0, 1].
    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.
    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).
    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.
    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.
    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.
    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def crop_border(imgs, crop_border):
    """Crop borders of images.
    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.
    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]


# Reference from https://github.com/YasinEnigma/Image_Interpolation/blob/main/main.py
# Image_Interpolation Github
# Originally Written by YasinEnigma
def nearest_interpolation(image, dimension):
    '''Nearest neighbor interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension
    Returns:
    numpy.ndarray: Resized image
    '''
    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    enlarge_time = int(sqrt((dimension[0] * dimension[1]) / (image.shape[0] * image.shape[1]))) if dimension[0] > image.shape[0] else int(sqrt((image.shape[0] * image.shape[1]) / (dimension[0] * dimension[1])))

    for i in range(dimension[0]):
        for j in range(dimension[1]):
            row = floor(i / enlarge_time)
            column = floor(j / enlarge_time)

            new_image[i, j] = image[row, column]

    return new_image


def bilinear_interpolation(image, dimension):
    '''Bilinear interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension
    Returns:
    numpy.ndarray: Resized image
    '''
    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    for k in range(3):
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                x = (j+0.5) * (scale_x) - 0.5
                y = (i+0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, width-2)
                y_int = min(y_int, height-2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int+1, k]
                c = image[y_int+1, x_int, k]
                d = image[y_int+1, x_int+1, k]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff)*(y_diff) + d*x_diff*y_diff

                new_image[i, j, k] = pixel.astype(np.uint8)

    return new_image


def W(x):
    '''Weight function that return weight for each distance point
    Parameters:
    x (float): Distance from destination point
    Returns:
    float: Weight
    '''
    a = -0.5
    pos_x = abs(x)
    if -1 <= abs(x) <= 1:
        return ((a+2)*(pos_x**3)) - ((a+3)*(pos_x**2)) + 1
    elif 1 < abs(x) < 2 or -2 < x < -1:
        return ((a * (pos_x**3)) - (5*a*(pos_x**2)) + (8 * a * pos_x) - 4*a)
    else:
        return 0


def bicubic_interpolation(img, dimension):
    '''Bicubic interpolation method to convert small size image to original size image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension
    Returns:
    numpy.ndarray: Resized image
    '''
    nrows = dimension[0]
    ncols = dimension[1]

    output = np.zeros((nrows, ncols, img.shape[2]), np.uint8)
    for c in range(img.shape[2]):
        for i in range(nrows):
            for j in range(ncols):
                xm = (i + 0.5) * (img.shape[0]/dimension[0]) - 0.5
                ym = (j + 0.5) * (img.shape[1]/dimension[1]) - 0.5

                xi = floor(xm)
                yi = floor(ym)

                u = xm - xi
                v = ym - yi

                # -------------- Using this make ignore some points and increase the value of black in image border
                # x = [(xi - 1), xi, (xi + 1), (xi + 2)]
                # y = [(yi - 1), yi, (yi + 1), (yi + 2)]
                # if ((x[0] >= 0) and (x[3] < img.shape[1]) and (y[0] >= 0) and (y[3] < img.shape[0])):
                #     dist_x0 = W(x[0] - xm)
                #     dist_x1 = W(x[1] - xm)
                #     dist_x2 = W(x[2] - xm)
                #     dist_x3 = W(x[3] - xm)
                #     dist_y0 = W(y[0] - ym)
                #     dist_y1 = W(y[1] - ym)
                #     dist_y2 = W(y[2] - ym)
                #     dist_y3 = W(y[3] - ym)

                #     out = (img[x[0], y[0], c] * (dist_x0 * dist_y0) +
                #            img[x[0], y[1], c] * (dist_x0 * dist_y1) +
                #            img[x[0], y[2], c] * (dist_x0 * dist_y2) +
                #            img[x[0], y[3], c] * (dist_x0 * dist_y3) +
                #            img[x[1], y[0], c] * (dist_x1 * dist_y0) +
                #            img[x[1], y[1], c] * (dist_x1 * dist_y1) +
                #            img[x[1], y[2], c] * (dist_x1 * dist_y2) +
                #            img[x[1], y[3], c] * (dist_x1 * dist_y3) +
                #            img[x[2], y[0], c] * (dist_x2 * dist_y0) +
                #            img[x[2], y[1], c] * (dist_x2 * dist_y1) +
                #            img[x[2], y[2], c] * (dist_x2 * dist_y2) +
                #            img[x[2], y[3], c] * (dist_x2 * dist_y3) +
                #            img[x[3], y[0], c] * (dist_x3 * dist_y0) +
                #            img[x[3], y[1], c] * (dist_x3 * dist_y1) +
                #            img[x[3], y[2], c] * (dist_x3 * dist_y2) +
                #            img[x[3], y[3], c] * (dist_x3 * dist_y3))

                #     output[i, j, c] = np.clip(out, 0, 255)
                # ---------------------------

                out = 0
                for n in range(-1, 3):
                    for m in range(-1, 3):
                        if ((xi + n < 0) or (xi + n >= img.shape[1]) or (yi + m < 0) or (yi + m >= img.shape[0])):
                            continue

                        out += (img[xi+n, yi+m, c] * (W(u - n) * W(v - m)))

                output[i, j, c] = np.clip(out, 0, 255)

    return output

# Reference from https://github.com/megvii-research/DCLS-SR
# Deep Constrained Least Squares for Blind Image Super-Resolution, https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Deep_Constrained_Least_Squares_for_Blind_Image_Super-Resolution_CVPR_2022_paper.pdf
# Originally Written by Ziwei Luo, Reference by Daejune Choi.


# fft 옛날 버전으로, 현재로써는 변환 필요

# ------------------------------------------------------
# -----------Constraint Least Square Filter-------------
def get_uperleft_denominator(img, kernel, grad_kernel):
    ker_f = convert_psf2otf(kernel, img.size()) # discrete fourier transform of kernel
    ker_p = convert_psf2otf(grad_kernel, img.size()) # discrete fourier transform of kernel

    denominator = inv_fft_kernel_est(ker_f, ker_p)
    numerator = torch.fft.fft2(img)
    if torch.isnan(numerator).any():
        print(f'numerator : {numerator}')
        assert False, 'numerator is nan'  
    deblur = deconv(denominator, numerator)
    deblur = torch.abs(deblur)
    
    return deblur

# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, ker_p):
    inv_denominator = ker_f.real * ker_f.real \
                      + ker_f.imag * ker_f.imag \
                      + ker_p.real * ker_p.real \
                      + ker_p.imag * ker_p.imag
    if torch.isnan(inv_denominator).any():
        print(f'inv_denominator : {inv_denominator}')
        assert False, 'inv_denominator is nan'  
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f).cuda()
    inv_ker_f.real = ker_f.real / inv_denominator
    inv_ker_f.imag = -ker_f.imag / inv_denominator
    if torch.isnan(inv_ker_f).any():
        print(f'inv_ker_f : {inv_ker_f}')
        assert False, 'inv_ker_f is nan'  
    return inv_ker_f

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f.real = inv_ker_f.real * fft_input_blur.real \
                            - inv_ker_f.imag * fft_input_blur.imag
    deblur_f.imag = inv_ker_f.real * fft_input_blur.imag \
                            + inv_ker_f.imag * fft_input_blur.real
    if torch.isnan(deblur_f).any():
        print(f'deblur_f : {deblur_f}')
        assert False, 'deblur_f is nan'  
    deblur = torch.fft.ifft2(deblur_f)
    if torch.isnan(deblur).any():
        print(f'deblur : {deblur}')
        assert False, 'deblur is nan'  
    return deblur


# --------------------------------
# --------------------------------
def convert_psf2otf(ker, size):
    psf = torch.zeros(size).cuda()

    # circularly shift
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]

    # compute the otf
    otf = torch.fft.fft2(psf)
    if torch.isnan(otf).any():
        print(f'otf : {otf}')
        assert False, 'otf is nan'    
    return otf



def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round
    is_numpy = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.transpose(2, 0, 1))
        is_numpy = True
    device = img.device
    # device = torch.device("cuda")

    is_batch = True
    if len(img.shape) == 3: # C, H, W
        img = img[None]
        is_batch = False

    B, in_C, in_H, in_W = img.size()
    img = img.view(-1, in_H, in_W)
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    # print(weights_H.device, indices_H.device, device)
    weights_H, indices_H = weights_H.to(device), indices_H.to(device)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W = weights_W.to(device), indices_W.to(device)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(B*in_C, in_H + sym_len_Hs + sym_len_He, in_W).to(device)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(B*in_C, out_H, in_W).to(device)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[:, i, :] = (img_aug[:, idx:idx + kernel_width, :].transpose(1, 2).matmul(
            weights_H[i][None,:,None].repeat(B*in_C,1, 1))).squeeze()

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(B*in_C, out_H, in_W + sym_len_Ws + sym_len_We).to(device)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(B*in_C, out_H, out_W).to(device)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, :, i] = (out_1_aug[:, :, idx:idx + kernel_width].matmul(
            weights_W[i][None,:,None].repeat(B*in_C, 1, 1))).squeeze()

    out_2 = out_2.contiguous().view(B, in_C, out_H, out_W)
    if not is_batch:
        out_2 = out_2[0]
    return out_2.cpu().numpy().transpose(1, 2, 0) if is_numpy else out_2

# Functions
# matlab 'imresize' function, now only support 'bicubic'

def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3

    weight = (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))
    return weight