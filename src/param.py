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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_PATH = "./log/"
LOG_NAME = strftime("%Y-%m-%d_%H-%M/", localtime())

# System
# Network

COMM = "fake_socket"
IP_ADDRESS = "127.0.0.1"
IP_PORT = "1141"

BUFFER_SIZE = 8192
DIST_BACKEND = "gloo"

#   Training hyper parameters
DATASET = "MNIST"
FOLDER = "iid_10"

MODEL = "MLP"
MODEL_PARAM = {"input_size":784, "output_size": 10, "channel": 1}

N_NODES = 11
N_EPOCH = 1

# CRITERION = "CrossEntropy"
CRITERION = "CrossEntropy"
OPTIMIZER = "SGD"
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.005

# Attack Settings
BAD_CLIENTS = [1]

# LDP-FL
N_ROUND = 20
KAP = [10] * N_ROUND
LATENCY_T = 10
EPS = 8

# FedSel
CLIPSIZE = 1.0
