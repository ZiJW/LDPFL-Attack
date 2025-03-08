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
DEVICE = torch.device("cuda:1")
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
FOLDER = "iid_10_with_public"

MODEL = "MLP"
MODEL_PARAM = {"input_size":784, "output_size": 10, "channel": 1}

# MODEL = "LR"
# MODEL_PARAM = {"input_size":13, "output_size": 2, "channel": 1}

N_NODES = 11
N_EPOCH = 1

CRITERION = "CrossEntropy"
OPTIMIZER = "SGD"
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.03

# FL Settings
FL_RULE = "DPSGD"
N_ROUND = 60
KAP = [10] * N_ROUND
LATENCY_T = 10
CLIENTS_WEIGHTS = [0] + [2] + [2] * (N_NODES - 2)

# LDP Settings
LDP = True
EPS = 4
NORM_BOUND = 5.0
SIGMA = 5.0

# Attack Settings
BAD_CLIENTS = []

# FedSel
CLIPSIZE = 2.0

# DPSGD
USE_TRANSFORM = False
ADVERSARY_ITERATION = 20

# Multi-Krum
MKRUM = False  #whether use multi-krum to detect broken or bad client or not
BROKEN_CLIENTS = []  #broken clients, used for test whether krum work or not
MAX_FAILURE = 1
KRUM_SELECTED = 3