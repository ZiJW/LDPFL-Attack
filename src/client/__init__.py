import client.base_client
import client.FedAvg
import client.FedMD
import client.LDP_FL
import client.Test_FL
import client.PrivFL
import client.DP_SGD

FedAvg_client = client.FedAvg.FedAvg_client
FedMD_client = client.FedMD.FedMD_client
LDPFL_client = client.LDP_FL.LDPFL_client
PrivFL_client = client.PrivFL.PrivFL_client
Test_client = client.Test_FL.Test_client
DPSGD_client = client.DP_SGD.DPSGD_client