import numpy as np


DATASET = 'DFEW'
#TRAIN_DIR = '/data/nyud'  # 'Modify data path'
TRAIN_DIR = './DFEW4ABAW40713'#'./DFEW4ABAW'  # 'Modify data path'
TRAIN_DIR_va = './RGB'
VAL_DIR = './FACE4ABAWSPLIT'#'./DFEW4ABAW40713/FACE4ABAWSPLIT'
TEST_DIR='./FACE4ABAWSPLIT2'
TRAIN_LIST = 'ABAWselect4refine0714.csv'#'ABAWselect3refineTRAIN.csv'
TRAIN_LIST_va = 'DFEWVALABEL.csv'
VAL_LIST = 'ABAWVAL712.csv'#'ABAWselect4refine0714.csv'DFEW4ABAW40713
TEST_LIST = 'ABAWVAL716.csv'


SHORTER_SIDE = 224
CROP_SIZE = 224
RESIZE_SIZE = None

NORMALISE_PARAMS = [1./255,  # Image SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # Image MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),  # Image STD
                    1./5000]  # Depth SCALE
BATCH_SIZE = 90
NUM_WORKERS = 3
NUM_CLASSES = 7
LOW_SCALE = 0.5
HIGH_SCALE = 2.0
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '18'  # ResNet101
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
FREEZE_BN = False
NUM_SEGM_EPOCHS = [50] * 2  # [150] * 3 if using ResNet152 as backbone
PRINT_EVERY = 1
RANDOM_SEED = 42
VAL_EVERY = 1  # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-4, 5e-4, 5e-4]#]#[5e-4, 5e-3, 5e-2]  # TO FREEZE, PUT 03e-4
LR_DEC = [5e-1, 1.5e-3, 7e-4]
MOM_ENC = 0.9  # TO FREEZE, PUT 0
MOM_DEC = 0.9
WD_ENC = 1e-5  # TO FREEZE, PUT 0
WD_DEC = 1e-5
LAMDA = 2e-4#2e-4  # slightly better
BN_threshold = 1e-1  # slightly better
OPTIM_DEC = 'sgd'
