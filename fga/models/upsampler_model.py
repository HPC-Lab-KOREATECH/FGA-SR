import tqdm
from copy import deepcopy
from os import path as osp
from collections import OrderedDict

import torch
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric

from basicsr.utils import get_root_logger
from basicsr.utils.img_util import imwrite, tensor2img

from basicsr.models.sr_model import BaseModel

from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class UpsamplerModel(BaseModel):
    """Upsampler model for single image super-resolution."""
    def __init__(self, opt):
        super(UpsamplerModel, self).__init__(opt)
        # data normalization
        self.data_range = self.opt['data_range'] if 'data_range' in opt else 1

        self.data_scaler = self.get_data_scaler(self.data_range)
        self.data_inverse_scaler = self.get_data_inverse_scaler(self.data_range)

        # define network
        self.net_g = build_network(opt['network_g']).to(self.device)
        self.net_upsampler = build_network(opt['network_upsampler']).to(self.device)

        self._inject_upsampler(self.net_upsampler, self.net_g)

        self.backbone_freeze = self.opt['backbone_freeze'] if 'backbone_freeze' in self.opt else True

        if not self.backbone_freeze:
            self.net_g = self.model_to_device(self.net_g)
        else:
            self.net_upsampler = self.model_to_device(self.net_upsampler)

        # load pretrained models
        load_g_path = self.opt['path'].get('pretrain_network_g', None)
        if load_g_path is not None:
            self.param_key = self.opt['path'].get('param_key_g', 'params')
            if self.param_key == 'None':
                self.param_key = None
            self.load_network(self.net_g, load_g_path, self.opt['path'].get('strict_load_g', True), self.param_key)

        self.print_network(self.net_upsampler)
        self.print_network(self.net_g)

        logger = get_root_logger()
        logger.info(f'Using data range: {self.data_range}')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_upsampler_ema = build_network(self.opt['network_upsampler']).to(self.device)
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            self._inject_upsampler(self.net_upsampler_ema, self.net_g_ema)

            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None and self.param_key is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight

            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('frequency_opt'):
            self.cri_frequency = build_loss(train_opt['frequency_opt']).to(self.device)
        else:
            self.cri_frequency = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_frequency is None:
            raise ValueError('Pixel and perceptual and frequency losses are None.')

        if self.backbone_freeze:
            for param in self.net_g.parameters():
                param.requires_grad = False
            for param in self.net_upsampler.parameters():
                param.requires_grad = True

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')

        if not self.backbone_freeze:
            self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
            self.optimizers.append(self.optimizer_g)
        else:
            self.optimizer_g = self.get_optimizer(optim_type, self.net_upsampler.parameters(), **train_opt['optim_g'])
            self.optimizers.append(self.optimizer_g)

    def _inject_upsampler(self, net_upsampler, net_g):
        replaced = False

        # Check and replace the 'upsample' attribute for IGNN, Swinir, CAT, DAT, HAT, DRCT
        if hasattr(net_g, 'upsample'):
            net_g.upsample = net_upsampler
            net_g.conv_last = torch.nn.Identity()
            replaced = True

        # Check and replace the 'tail' attribute for EDSR, RCAN, SAN, NLSN, HAN, IGNN, IPT. Because of the IGNN.
        # In this x4 case, IGNN have upsample and tail
        if hasattr(net_g, 'tail'):
            if isinstance(net_g.tail, torch.nn.ModuleList): # Because of the IPT
                net_g.tail[0][-2] = net_upsampler
                net_g.tail[0][-1] = torch.nn.Identity()
            else:
                net_g.tail[-2] = net_upsampler
                net_g.tail[-1] = torch.nn.Identity()
            replaced = True

        # Check and replace scale-specific tail attributes for EDT
        scale_attr = f'tail_sr_x{self.opt["scale"]}'
        if hasattr(net_g, scale_attr):
            tail = getattr(net_g, scale_attr)
            if isinstance(tail, torch.nn.ModuleList):
                tail[-2] = net_upsampler
                tail[-1] = torch.nn.Identity()
                replaced = True

        if not replaced:
            raise ValueError('Upsampler not found in net_g.')

    def feed_data(self, data):
        self.lq = self.data_scaler(data['lq'].to(self.device))
        if 'gt' in data:
            self.gt = self.data_scaler(data['gt'].to(self.device))

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # frequency loss
        if self.cri_frequency:
            l_freq = self.cri_frequency(self.output, self.gt)
            if l_freq is not None:
                l_total += l_freq
                loss_dict['l_freq'] = l_freq

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def get_data_scaler(self, data_range):
        """Data normalizer. Assume data are always in [0, 1]."""
        if data_range == 255:
            # Rescale to [0, 255]
            return lambda x: x * 255.
        else:
            return lambda x: x


    def get_data_inverse_scaler(self, data_range):
        """Inverse data normalizer."""
        if data_range == 255:
            # Rescale [0, 255] to [0, 1]
            return lambda x: x / 255.
        else:
            return lambda x: x


    def test(self):
        scale = self.opt.get('scale', 1)
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size'] if 'window_size' in self.opt['network_g'] else 1
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.data_inverse_scaler(self.lq.detach().cpu())
        out_dict['result'] = self.data_inverse_scaler(self.output.detach().cpu())
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.data_inverse_scaler(self.gt.detach().cpu())
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)