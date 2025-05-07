import server.FedAvg
import server.FedMD
import server.LDP_FL
import server.Test_FL
import server.PrivFL
import server.DP_SGD

FedAvg_server = server.FedAvg.FedAvg_server
FedMD_server = server.FedMD.FedMD_server
LDPFL_server = server.LDP_FL.LDPFL_server
MKrum_server = server.Test_FL.Test_server
PrivFL_server = server.PrivFL.PrivFL_server
DPSGD_server = server.DP_SGD.DPSGD_server