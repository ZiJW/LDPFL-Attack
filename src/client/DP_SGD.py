import torch
import logging
from geom_median.torch import compute_geometric_median 
from base_client import Base_client
from model import load_model, load_criterion, load_optimizer
from util import load_dataset, ExpM, pm_perturbation
from param import DEVICE
import param
from tqdm import tqdm

class Transf(torch.nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.params = torch.nn.ParameterDict({
            'alpha': torch.nn.Parameter(torch.rand(input_size)),
            'beta': torch.nn.Parameter(torch.rand(input_size))
        })

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return torch.add(torch.mul(x, self.params['alpha']), self.params['beta'])

class DPSGD_client(Base_client):  
    """
        The client in FedSel
    """

    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM,
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)[0]
        self.valid_loader = load_dataset(param.DATASET, param.FOLDER, [("public", False)])[0] if id in param.BAD_CLIENTS else []
        self.round = param.N_ROUND
        self.privacy1_percent = 0.1
        self.layer0 = Transf(param.MODEL_PARAM["input_size"]).to(param.DEVICE)
        self.ldp = param.LDP and self.id not in param.BAD_CLIENTS

        if self.id in param.BAD_CLIENTS:
            logging.info("Client {} is an adversary!".format(self.id))

    def train(self, rn):
        """
            Training for 1 round.
        """
        logging.debug("Client {}: wait for invitation ...".format(self.id))

        chose = self.comm.recv(0)
        if not chose:
            self.comm.send(0, "ACK")
            return
        
        logging.debug("Client {} : receive invitation {}".format(self.id, chose))
        self.comm.send(0, "ACK")

        # Download the global model parameters
        global_model = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server: {}".format(self.id, global_model))
        self.unserialize_model(global_model)

        torch.autograd.set_detect_anomaly(True)
        gradients = torch.tensor([0.]*len(global_model)).to(param.DEVICE)
        
        if self.id in param.BAD_CLIENTS and param.ADVERSARY_ITERATION != 0:
            est_good_gradients = self.fit_on(self.train_loader, len(global_model))[1].mul((param.N_NODES-2)/(param.N_NODES-1))
            self.unserialize_model(global_model-est_good_gradients*param.LEARNING_RATE)
            std_acc = self.test_on_public(self.model)[0]*0.9

            for i in range(param.ADVERSARY_ITERATION):
                est_gradients = est_good_gradients+gradients.div(param.N_NODES-1)
                self.unserialize_model(global_model-est_gradients*param.LEARNING_RATE)
                bad_gradients = self.fit_on(self.train_loader, len(global_model), rev=True)[1]
                gradients += bad_gradients
                for j in range(10):
                    self.unserialize_model(global_model-gradients*param.LEARNING_RATE)
                    acc = self.test_on_public(self.model)[0]
                    if acc > std_acc:
                        break
                    valid_set_gradients = self.fit_on(self.valid_loader, len(global_model))[1]
                    gradients += valid_set_gradients

                acc = self.test_on_public(self.model)[0]
                logging.info("Client {}: iteration {}, valid set acc = {}, std-acc : {}".format(self.id, i, acc, std_acc))
            return_gradients = gradients
        elif self.id in param.BAD_CLIENTS:
            logging.debug("Client {}: iteration {}, generate adversarial gradients".format(self.id, rn))
            layer0_grads, train_set_gradients = self.fit_on(self.train_loader, len(global_model), rev=True, transform=param.USE_TRANSFORM)
            res = []
            for val in self.layer0.state_dict().values():
                res.append(val.view(-1))
            res = torch.cat(res)
            self.unserialize_model(res-layer0_grads*param.LEARNING_RATE, self.layer0)
            self.unserialize_model(self.serialize_model()- train_set_gradients * param.LEARNING_RATE * param.ADVERSARY_SCALE)
            return_gradients = train_set_gradients * param.ADVERSARY_SCALE
        else:
            layer0_grads, train_set_gradients = self.fit_on(self.train_loader, len(global_model), transform=param.USE_TRANSFORM)
            res = []
            for val in self.layer0.state_dict().values():
                res.append(val.view(-1))
            res = torch.cat(res)
            self.unserialize_model(res-layer0_grads*param.LEARNING_RATE, self.layer0)
            self.unserialize_model(self.serialize_model()-train_set_gradients*param.LEARNING_RATE)
            return_gradients = train_set_gradients

        logging.debug("CLient {} : gradients : {}".format(self.id, return_gradients))
        # logging.debug("Client {}: selected_index = {}, max absval = {}".format(self.id, selected_index, torch.max(accum_grad)))
        return return_gradients

    def fit_on(self, loader, model_size, rev = False, transform=False):
        gradients = torch.tensor([0.]*model_size).to(param.DEVICE)
        layer0_grads = torch.tensor([0.]*(2*param.MODEL_PARAM["input_size"])).to(param.DEVICE)
        cnt_samples = 0
        for data, target in loader:
            # Training
            # logging.debug("Client {}: training in batch {} with optimer={} ...".format(self.id, batch_idx, type(self.optimizer)))
            data, target = data.to(DEVICE), target.to(DEVICE)
            self.optimizer.zero_grad()
            output = self.model(self.layer0(data)) if transform is True else self.model(data)
            # logging.debug("output = {}".format(output))
            # logging.debug("target = {}".format(target))
            loss = -self.criterion(output, target) if rev else self.criterion(output, target)
            loss.backward()
            
            gradient = []
            for val in self.model.parameters():
                gradient.append(val.grad.view(-1))
            gradient = torch.cat(gradient)
            norm = torch.sqrt(torch.sum(torch.pow(gradient, 2)))
            if norm > param.NORM_BOUND:
                gradient.div_(norm.div(param.NORM_BOUND))
            #noise = self.GaussianNoise(param.SIGMA, param.NORM_BOUND, gradients.shape)
            #logging.debug("Client {} norm : {} parameter : {} with noise : {}".format(self.id, norm, gradients, noise))
            #gradient += noise
            gradient.div_(len(data))
            gradients += gradient
            cnt_samples += len(data)
        
            if transform is True:
                gradient = []
                for val in self.layer0.parameters():
                    gradient.append(val.grad.view(-1))
                gradient = torch.cat(gradient)
                norm = torch.sqrt(torch.sum(torch.pow(gradient, 2)))
                if norm > param.NORM_BOUND:
                    gradient.div_(norm.div(param.NORM_BOUND))
                gradient.div_(len(data))
                layer0_grads += gradient
        if self.ldp:
            noise = self.GaussianNoise(param.SIGMA, param.NORM_BOUND, gradients.shape) / cnt_samples
            logging.debug("Client {} norm : {} parameter : {} with noise : {}".format(self.id, norm, gradients, noise))
            gradients += noise

        return layer0_grads, gradients

    def test_on_public(self, model):
        """
            Test the accuracy and loss on validation dataset.
        """
        Acc, Loss = 0, 0.0
        LossList = []
        N = 0
        for data, target in self.valid_loader:
            with torch.no_grad():
                data, target = data.to(param.DEVICE), target.to(param.DEVICE)
                output = model(data)
                loss = self.criterion(output, target)
                _, pred = torch.max(output, dim=1)
                
                Loss += loss.item()
                LossList.append(loss.item())
                Acc += torch.sum(pred.eq(target)).item()
                N += len(target)

        Acc = Acc / N
        Loss = Loss / len(self.valid_loader)
        return Acc, Loss

    def send2server(self, msg):
        self.comm.send(0, msg)
        assert self.comm.recv(0) == "ACK"
        logging.debug("Client {}: send local grads to server".format(self.id))

    def receive_global_model(self):
        logging.debug("Client {}: wait for invitation ...".format(self.id))

        chose = self.comm.recv(0)
        if not chose:
            self.comm.send(0, "ACK")
            return
        
        logging.debug("Client {} : receive invitation {}".format(self.id, chose))
        self.comm.send(0, "ACK")

        # Download the global model parameters
        global_model = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server: {}".format(self.id, global_model))
        return global_model

    def receive_other_parameter(self):
        record_param = self.comm.recv(0)
        logging.debug("Client {} got other clients' parameter from server".format(self.id))
        param_list = [value for idx, value in record_param]
        return param_list
    
    def generate_fit_gradients(self, global_model, gradient_list, rn):
        """
        训练 self.model 使其沿 loss 取反方向优化 并同时接近中心点  (median ) 
        最终输出相对于 global_model 的梯度
        """
        # 计算几何中位数（gradient_list 中的几何中心点）
        median = compute_geometric_median(gradient_list).median
        closest_gradient = min(gradient_list, key=lambda g: torch.norm(g - median))
        # 允许的偏移范围是最近点到中心点之间的差值*lr，这表示两个对应模型之间的距离
        epsilon = torch.norm(param.LEARNING_RATE * (closest_gradient - median), p=2) 

        self.unserialize_model(global_model - param.LEARNING_RATE * median)

        theta_target = self.serialize_model()  # 目标中心点
        lambda_reg = param.LAMBDA_REG  # 正则项权重
        k = param.HINGE_POWER  # 控制增长速率（通常 k=2）

        # 训练 self.model，使其沿 loss 取反方向优化
        for i in  tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(self.layer0(data)) if param.USE_TRANSFORM else self.model(data)

                # 计算 loss（取反方向优化）
                loss = -self.criterion(output, target)  # 逆向优化

                # 计算指数正则项
                theta = self.serialize_model()
                norm_diff = torch.norm(theta - theta_target, p=2)
                reg_loss = lambda_reg * torch.pow(torch.clamp(norm_diff - epsilon, min=0), k)

                # 总损失
                total_loss = loss + reg_loss
                total_loss.backward()
                for params in self.model.parameters():
                    if params.grad is not None:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)  # 分层裁剪

                self.optimizer.step()
            logging.debug("Client {} : fit adversarial training round {}, loss1 : {}, reg_loss : {}".format(self.id, i, loss, reg_loss))

        updated_model_params = self.serialize_model()
        gradients = (global_model - updated_model_params) / param.LEARNING_RATE

        return gradients

    def generate_normed_gradients(self, global_model, gradient_list, rn):
        """
        训练 self.model 使其沿 loss 取反方向优化 并同时接近中心点  (median ) 
        最终输出相对于 global_model 的梯度
        """
        # 计算几何中位数（gradient_list 中的几何中心点）
        median = compute_geometric_median(gradient_list).median
        reference_gradient = min(gradient_list, key=lambda g: torch.norm(g - median))
        # 允许的偏移范围是最近点到中心点之间的差值*lr，这表示两个对应模型之间的距离
        C = param.ADVERSARY_SCALE * torch.norm(param.LEARNING_RATE * (reference_gradient - median), p=2) 

        self.unserialize_model(global_model - param.LEARNING_RATE * median)
        theta_target = self.serialize_model()  # 目标中心点
        logging.debug("Client {} C {} parameter {}".format(self.id, C, theta_target))

        # 训练 self.model，使其沿 loss 取反方向优化
        for i in tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)

                # 计算 loss（取反方向优化）
                loss = -self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            #norm clip
            theta = self.serialize_model()
            logging.debug("Client {} round {}.{} parameter {}".format(self.id, rn, i, theta))
            diff = theta - theta_target
            rate = max(1.0, torch.norm(diff, p=2) / (C))
            clipped_theta = theta_target + diff / rate
            self.unserialize_model(clipped_theta)
            logging.debug("Client {} round {}.{} rate {} parameter {}".format(self.id, rn, i, rate, clipped_theta))


        updated_model_params = self.serialize_model()
        true_gradients = (global_model - updated_model_params) / param.LEARNING_RATE

        return true_gradients

    def generate_clip_gradients(self, global_model, gradient_list, rn):
        good_gradient_stack = torch.stack(gradient_list)
        mean = torch.mean(good_gradient_stack, dim = 0)
        self.unserialize_model(global_model - param.LEARNING_RATE * mean)
        sorted_vals, _ = torch.sort(good_gradient_stack, dim=0)  # shape: [num_clients, num_params]
        beta = param.TRIMMED_MEAN_BETA
        select = sorted_vals[int(beta * param.ADVERSARY_SCALE):-int(beta * param.ADVERSARY_SCALE)]
        clip_mini, clip_maxi = select.min(dim=0).values, select.max(dim=0).values
        # 训练 self.model，使其沿 loss 取反方向优化
        for i in tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)

                # 计算 loss（取反方向优化）
                loss = -self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            theta = self.serialize_model()
            gradient = (global_model - theta) / param.LEARNING_RATE
            gradient_clip = torch.clamp(gradient, min=clip_mini, max=clip_maxi)
            self.unserialize_model(global_model - param.LEARNING_RATE * gradient_clip)

        updated_model_params = self.serialize_model()
        true_gradients = (global_model - updated_model_params) / param.LEARNING_RATE
        return true_gradients

    def evaluate(self):
        """
            Train the model on all rounds.
        """
        self.comm.initialize()
        for rn in range(self.round):
            if self.id in param.TAPPING_CLIENTS :
                global_model = self.receive_global_model()
                good_gradients = self.receive_other_parameter()
                if param.MKRUM:
                    #bad_gradient = self.generate_fit_gradients(global_model, good_gradients, rn)
                    bad_gradient = self.generate_normed_gradients(global_model, good_gradients, rn)
                elif param.TRIMMED_MEAN:
                    bad_gradient = self.generate_clip_gradients(global_model, good_gradients, rn)
                else:
                    assert(0)
                logging.debug("Client {} bad model parameter shape : {}, other model parameter shape : {}".format(self.id, bad_gradient.shape, good_gradients[0].shape))
                self.send2server(bad_gradient)
            else :
                res = self.train(rn)
                self.send2server(res)