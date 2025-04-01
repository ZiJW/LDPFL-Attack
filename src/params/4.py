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
IP_PORT = "11417"

BUFFER_SIZE = 8192
DIST_BACKEND = "gloo"

#   Training hyper parameters
# DATASET = "adult"
# FOLDER = "iid_30"
DATASET = "MNIST"
FOLDER = "dirichlet_20users_a1000000.0_seed98_public0.05"

MODEL = "VGG_Mini"
MODEL_PARAM = {"input_size":784, "output_size": 10, "channel": 1}

# MODEL = "MLP"
# MODEL_PARAM = {"input_size":13, "output_size": 2, "channel": 1}

N_NODES = 21
N_EPOCH = 1

CRITERION = "CrossEntropy"
OPTIMIZER = "SGD"
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.03

# FL Settings
FL_RULE = "DPSGD"
N_ROUND = 100
KAP = [N_NODES - 1] * N_ROUND

# LDP Settings
LDP = True
EPS = 0.72
LATENCY_T = 10
CLIENTS_WEIGHTS = [0] + [2] + [2] * (N_NODES - 2)

# FedSel
CLIPSIZE = 2.0

# DPSGD
USE_TRANSFORM = False
NORM_BOUND = 5.0
SIGMA = 90.0

# Multi-Krum
MKRUM = True
BROKEN_CLIENTS = []  #broken clients, used for test whether krum work or not
MAX_FAILURE = 8
KRUM_SELECTED = 10

# Attack Settings
BAD_CLIENTS = [1, 2]
ADVERSARY_ITERATION = 100
ADVERSARY_SCALE = 1.7
TAPPING_CLIENTS = BAD_CLIENTS
#TAPPING_CLIENTS = []
TAPPING_SAME = True