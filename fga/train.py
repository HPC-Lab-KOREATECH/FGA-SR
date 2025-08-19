import os.path as osp

import fga.archs
import fga.data
import fga.models
import fga.losses
import fga.metrics

from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)