# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import logging
import torch
from os import path as osp
import random
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs, set_random_seed)
from basicsr.utils.options import dict2str, parse
from basicsr.utils.dist_util import get_dist_info

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse(opt_path="options/test/Rain13k/Wavelet-width32.yml", is_train=False)
    opt['dist'] = False
    opt['rank'], opt['world_size'] = get_dist_info()
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    # log_file = osp.join(opt['path']['log'], opt['name']+"_"+opt['path']["pretrain_network_g"].split("/")[-1].split('.')[0],
    #                     f"{opt['datasets']['val']['dataroot_gt'].split('/')[-2]}.log")
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    print(f"log file path: {log_file}")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    print("logger loaded")
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'test' in phase:
            dataset_opt['phase'] = 'test'
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)
    print("model loaded!")
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        psnr = model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr, use_image=use_image)
    return float(psnr)

# if __name__ == '__main__':
#     main()
    # print("I am running")