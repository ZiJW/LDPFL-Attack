import torch
import sys
from time import strftime, localtime

sys.path.append("./dataset")
sys.path.append("./server")
sys.path.append("./client")
sys.path.append("./network")
sys.path.append("./model")
sys.path.append("./test")

# System
SEED = 114
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_PATH = "/home/wzj/FLattack/FedField/log/"
LOG_NAME = strftime("%Y-%m-%d_%H-%M/", localtime())

# System
# Network

COMM = "fake_socket"
IP_ADDRESS = "127.0.0.1"
IP_PORT = "11418"

BUFFER_SIZE = 8192
DIST_BACKEND = "gloo"

#   Training hyper parameters
# DATASET = "adult"
#DATASET = "MNIST"
#DATASET = "FashionMNIST"
DATASET = "CIFAR10"
#FOLDER = "iid_10_with_public"
FOLDER = "dirichlet_20users_a500_seed98_public0.05"
DATA_AGUMENT = (DATASET == "CIFAR10")

MODEL = "VGG_Mini"
MODEL_PARAM = {"input_size":1024, "output_size": 10, "channel": 3}

# MODEL = "MLP"
# MODEL_PARAM = {"input_size":13, "output_size": 2, "channel": 1}

# FL Settings
N_NODES = 21
FL_RULE = "DPSGD"
N_ROUND = 100
N_EPOCH = 5
KAP = [N_NODES - 1] * N_ROUND

CRITERION = "CrossEntropy"
OPTIMIZER = "SGD"
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 100
LEARNING_RATE = 0.1
LEARNING_RATE_LIST = [0.1] * N_ROUND

# LDP Settings
LDP = True
EPS = 0.75
LATENCY_T = 10
CLIENTS_WEIGHTS = [0] + [2] + [2] * (N_NODES - 2)

# FedSel
CLIPSIZE = 2.0

# DPSGD
USE_TRANSFORM = False
NORM_BOUND = 4.0
SIGMA = 0.5
DELTA = 1e-4
P = 0.1

# Multi-Krum
MKRUM = False
BROKEN_CLIENTS = []  #broken clients, used for test whether krum work or not
MAX_FAILURE = 8
KRUM_SELECTED = 10


#Trimmed Mean
TRIMMED_MEAN = False
TRIMMED_MEAN_BETA = int((N_NODES - 1) / 4)

# Attack Settings
BAD_CLIENTS = [1, 2]
#TAPPING_CLIENTS = BAD_CLIENTS
TAPPING_CLIENTS = []
ATTACK_MODE = "back-total-loss"
ADVERSARY_ITERATION = [10] * N_ROUND
ADVERSARY_SCALE = [1.0] * N_ROUND
ADVERSARY_NORM = 5.0
TAPPING_SAME = True