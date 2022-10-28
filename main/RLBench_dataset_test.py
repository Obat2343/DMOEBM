from distutils.log import debug
import os 
import sys
import time
import shutil
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
from pycode.config import _C as cfg
cfg.DATASET.RLBENCH.TASK_NAME = ['PickUpCup']

from pycode.dataset import RLBench_dataset_IBC

##### config #####
num_trial = cfg.OUTPUT.NUM_TRIAL
max_iter = cfg.OUTPUT.MAX_ITER
save_iter = cfg.OUTPUT.SAVE_ITER
log_iter = cfg.OUTPUT.LOG_ITER

batch_size = cfg.DATASET.BATCH_SIZE
image_size = cfg.DATASET.IMAGE_SIZE
save_dir = os.path.join(cfg.OUTPUT.BASE_DIR, cfg.DATASET.NAME, cfg.DATASET.RLBENCH.TASK_NAME)

dims = cfg.MODEL.DIMS
enc_depths = cfg.MODEL.ENC_DEPTHS
enc_layers = cfg.MODEL.ENC_LAYERS
dec_depths = cfg.MODEL.DEC_DEPTHS
dec_layers = cfg.MODEL.DEC_LAYERS
extractor_list = cfg.MODEL.EXTRACTOR_LIST
predictor_name = cfg.MODEL.PREDICTOR_NAME
heads = cfg.MODEL.ATTENTION.HEADS
num_vec = cfg.MODEL.SOFTARGMAX.NUM
query_list = cfg.DATASET.RLBENCH.QUERY_LIST
query_dims = cfg.DATASET.RLBENCH.QUERY_DIMS

positive_sample = cfg.SAMPLING.NUM_POSITIVE
device = 'cpu'

# make dataset
results = []
train_dataset = RLBench_dataset_IBC("val", cfg, save_dataset=True, debug=False)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

# start training
start = time.time()

for i, data in enumerate(dataloader):
    img, robot_obs, query, negative_sample, info = data

    # convert device
    img, robot_obs = img.to(device), robot_obs.to(device)
    for key in query.keys():
        query[key] = query[key].to(device)
    
    # for key in label.keys():
    #     label[key] = label[key].to(device)

    end = time.time()
    print("step:{} cost:{}".format(i, (end - start) / (i+1)))

    if i == max_iter:
        break
    