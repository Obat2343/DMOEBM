from yacs.config import CfgNode as CN
import os

_C = CN()

##################
##### OUTPUT ##### 
##################
_C.OUTPUT = CN()
_C.OUTPUT.BASE_DIR = "../result"
_C.OUTPUT.NUM_TRIAL = 5
_C.OUTPUT.MAX_ITER = 100000
_C.OUTPUT.SAVE_ITER = 10000 # interval to save model and log eval loss
_C.OUTPUT.LOG_ITER = 100 # interval to log training loss
_C.OUTPUT.EVAL_ITER = 1000

###################
##### DATASET ##### 
###################

_C.DATASET = CN()
_C.DATASET.NAME = "RLBench"
_C.DATASET.BATCH_SIZE = 32
_C.DATASET.IMAGE_SIZE = 256

### RLBENCH ###
_C.DATASET.RLBENCH = CN()
_C.DATASET.RLBENCH.TASK_NAME = 'PickUpCup' # e.g. 'CloseJar', 'PickUpCup'
_C.DATASET.RLBENCH.PATH = os.path.abspath('../dataset/RLBench-panda') # '../dataset/RLBench-Local'
_C.DATASET.RLBENCH.SEQ_LEN = 1000
_C.DATASET.RLBENCH.QUERY_LIST = ["uv", "time", "rotation_quat", "grasp", "z"]
_C.DATASET.RLBENCH.QUERY_DIMS = [2, 1, 4, 1, 1]

#################
##### NOISE #####
#################

_C.NOISE = CN()
_C.NOISE.METHOD = "gaussian" # gaussian, vae

### GAUSSIAN NOISE ###
_C.NOISE.GAUSSIAN = CN()
_C.NOISE.GAUSSIAN.POSE_RANGE = 0.03
_C.NOISE.GAUSSIAN.ROT_RANGE = 10.0
_C.NOISE.GAUSSIAN.GRASP_PROB = 0.1

### VAE NOISE ###
_C.NOISE.VAE = CN()
_C.NOISE.VAE.STD_MUL = 1.0

####################
##### SAMPLING #####
####################

_C.SAMPLING = CN()
_C.SAMPLING.FIRST_SAMPLE = "random_pick_with_noise" # random_range, random_pick, random_pick_with_noise, vae_noise
_C.SAMPLING.SECOND_SAMPLE = "DFO" # random_all, random_frame, langevin
_C.SAMPLING.INF_SAMPLE = "langevin"
_C.SAMPLING.NUM_NEGATIVE = 64
_C.SAMPLING.SECOND_SAMPLE_STEP = 0
_C.SAMPLING.LIMIT_SAMPLE = 8

### RANDOM_PICK_WITH_NOISE
_C.SAMPLING.RPWN = CN()
_C.SAMPLING.RPWN.RANGE = 0.1
_C.SAMPLING.RPWN.INCLUDE_SELF = True

### vae ###
_C.SAMPLING.VAE = CN()
_C.SAMPLING.VAE.NOISE_STD = 1.0

### langevin ###
_C.SAMPLING.LANGEVIN = CN()
_C.SAMPLING.LANGEVIN.STEP_SIZE = 0.01
_C.SAMPLING.LANGEVIN.MOMENTUM = 0.
_C.SAMPLING.LANGEVIN.ITERATION = 20
_C.SAMPLING.LANGEVIN.DECAY_RATIO = 0.5
_C.SAMPLING.LANGEVIN.DECAY_STEP = 5

### langevin_vae ###
_C.SAMPLING.LANGEVIN_VAE = CN()
_C.SAMPLING.LANGEVIN_VAE.STEP_SIZE = 0.01
_C.SAMPLING.LANGEVIN_VAE.MOMENTUM = 0.
_C.SAMPLING.LANGEVIN_VAE.ITERATION = 10
_C.SAMPLING.LANGEVIN_VAE.DECAY_RATIO = 0.5
_C.SAMPLING.LANGEVIN_VAE.DECAY_STEP = 5

### dfo ###
_C.SAMPLING.DFO = CN()
_C.SAMPLING.DFO.RATIO = 0.1
_C.SAMPLING.DFO.ITERATION = 20
_C.SAMPLING.DFO.DECAY_RATIO = 0.5
_C.SAMPLING.DFO.DECAY_STEP = 10

### dmo ###
_C.SAMPLING.DMO = CN()
_C.SAMPLING.DMO.ITERATION = 5
_C.SAMPLING.DMO.THRESHOLD = -10.
_C.SAMPLING.DMO.LIMIT_SAMPLE = 16
 
###################
###### MODEL ######
###################

_C.MODEL = CN()

_C.MODEL.CONV_DIMS = [96, 192, 384, 768]
_C.MODEL.ENC_LAYERS = ['convnext','convnext','convnext','convnext']
_C.MODEL.ENC_DEPTHS = [3,3,9,3]

_C.MODEL.DEC_LAYERS = ['convnext','convnext','convnext']
_C.MODEL.DEC_DEPTHS = [3,3,3]

_C.MODEL.EXTRACTOR_NAME = "query_uv_feature"
_C.MODEL.PREDICTOR_NAME = "Regressor_Transformer_with_cat_feature"

_C.MODEL.CONV_DROP_PATH_RATE = 0.1
_C.MODEL.ATTEN_DROPOUT_RATE = 0.1
_C.MODEL.QUERY_EMB_DIM = 128
_C.MODEL.NUM_ATTEN_BLOCK = 4

_C.VAE = CN()
_C.VAE.NAME = "VAE" # VAE, Transformer_VAE
_C.VAE.LATENT_DIM = 256
_C.VAE.KLD_WEIGHT = 0.01
###################
###### OPTIM ######
###################

_C.OPTIM = CN()

_C.OPTIM.LR = 1e-4

_C.OPTIM.SCHEDULER = CN()
_C.OPTIM.SCHEDULER.GAMMA = 0.99
_C.OPTIM.SCHEDULER.STEP = 1000