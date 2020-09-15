import numpy as np
import torch
from torch import nn
import time


epochs = 2000
SEED = 16
np.random.seed(SEED)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(SEED)


class CF(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num):
        super(CF, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = torch.nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                num_layers=layer_num)
        self.out = torch.nn.Linear(self.hidden_dim, 2)  # output layer

    def forward(self, miu):
        output, hn = self.gru(miu)
        output_in_last_timestep = hn[-1, :, :]
        x = self.out(output_in_last_timestep)
        return x


data_death = []
with open("./data/7/hadm_death_20_7.txt", "r") as file_death:
    for line in file_death.readlines():
        is_death = float(str(line.strip('\n')).split('\t')[1])
        data_death.append(is_death)
data_death_Tensor = torch.LongTensor(data_death)

group = 776
# cf_record = np.load("./data/7/CF_node_polynomial_eff_pro.npy")
# cf_record = np.load("./data/7/CF_node_is_fail.npy")
cf_record = np.load("./data/7/CF_node_eff_pro.npy")

hidden_dim_all = [32, 64, 128, 256]
weight_CRs = [0.3, 0.5, 0.7, 0.9, 1.0]
MAX_AUC_list = []
for index in range(5):
    # with open("./data/7/AUC_result_GRU_polynomial_eff_pro.txt", "a") as f:
    with open("./data/7/AUC_result_GRU_node_eff_pro.txt", "a") as f:
    # with open("./data/7/AUC_result_GRU_node_is_fail.txt", "a") as f:
        f.write(str(index+1)+'\n')
        max_AUC = 0
        for weight_CR in weight_CRs:
            for hidden_dim in hidden_dim_all: 
                length = 60
                input_dim = 451
                layer_num = 2
                learning_rate = 0.0005

                if index == 0:
                    cal_r_is_fail_train = cf_record[group * 1:]
                    cal_r_is_fail_test = cf_record[:group * 1]
                elif index == 4:
                    cal_r_is_fail_train = cf_record[:group * 4]
                    cal_r_is_fail_test = cf_record[group * 4:]
                else:
                    cal_r_is_fail_train = np.concatenate((cf_record[group * 0:group * index], cf_record[group * (index + 1):]))
                    cal_r_is_fail_test = cf_record[group * index:group * (index + 1)]

                miu_train_new = []
                for i in range(cal_r_is_fail_train.shape[1]):
                    miu_train_new.append(cal_r_is_fail_train[:, i, :])
                miu_train_new = torch.Tensor(miu_train_new)

                miu_test_new = []
                for i in range(cal_r_is_fail_test.shape[1]):
                    miu_test_new.append(cal_r_is_fail_test[:, i, :])
                miu_test_new = torch.Tensor(miu_test_new)

                net = CF(input_dim=input_dim, hidden_dim=hidden_dim, layer_num=layer_num)  # d
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
                loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor([weight_CR, 1.0]).cuda())

                if cuda:
                    net.cuda()
                    miu_train_new = miu_train_new.cuda()
                    miu_test_new = miu_test_new.cuda()
                    data_death_Tensor = data_death_Tensor.cuda()

                best_epoch = 0
                best_loss = 10000
                bad_count = 0
                patience = 100
                start = time.time()
                for epoch in range(epochs):
                    now = time.time()
                    net.train()
                    optimizer.zero_grad()
                    out_train = net(miu_train_new)

                    if index == 0:
                        loss = loss_func(out_train, data_death_Tensor[group*1:])
                    elif index == 4:
                        loss = loss_func(out_train, data_death_Tensor[:group*4])
                    else:
                        loss = loss_func(out_train, torch.cat((data_death_Tensor[:group*index], data_death_Tensor[group*(index+1):])))
                    loss.backward()
                    optimizer.step()

                    net.eval()
                    out_test = net(miu_test_new)
                    if index == 0:
                        loss_test = loss_func(out_test, data_death_Tensor[:group*1])
                        target_y = data_death_Tensor[:group*1].cpu().data.numpy()
                    elif index == 4:
                        loss_test = loss_func(out_test, data_death_Tensor[group*4:])
                        target_y = data_death_Tensor[group*4:].cpu().data.numpy()
                    else:
                        loss_test = loss_func(out_test, data_death_Tensor[group*index:group*(index+1)])
                        target_y = data_death_Tensor[group*index:group*(index+1)].cpu().data.numpy()

                    if loss_test < best_loss:
                        best_loss = loss_test
                        best_epoch = epoch
                        bad_count = 0
                        best_out_test = out_test.cpu()
                    else:
                        bad_count += 1

                    if bad_count == patience:
                        break

                True_sample_pro = best_out_test[:, 0].detach().numpy().tolist()
                index_dic = {}
                for id, pro in enumerate(True_sample_pro):
                    index_dic[id] = pro
                index_dic = sorted(index_dic.items(), key=lambda item:item[1], reverse=True)

                FPR_list = [0.0]
                TPR_list = [0.0]
                TP = 0
                FN = 0
                FP = 0
                TN = 0

                for id in index_dic:
                    if target_y[id[0]] == 0:
                        TP += 1
                    else:
                        FP += 1
                    if index == 4:
                        FN = group + 4 - target_y.sum() - TP
                    else:
                        FN = group - target_y.sum() - TP

                    TN = target_y.sum() - FP
                    FPR = FP / (TN + FP)
                    TPR = TP / (TP + FN)
                    FPR_list.append(FPR)
                    TPR_list.append(TPR)

                AUC = 0
                for id, x in enumerate(FPR_list[:-1]):
                    AUC += (FPR_list[id+1] - x) * (TPR_list[id] + TPR_list[id+1])
                AUC = AUC/2
                if AUC > max_AUC:
                    max_AUC = AUC

                f.write("seq_len=" + str(length) + '  ')
                f.write("input=" + str(input_dim) + '  ')
                f.write("hidden=" + str(hidden_dim) + '  ')
                f.write("layer_num=" + str(layer_num) + '  ')
                f.write("w_CR=" + str(weight_CR) + '  ')
                f.write("lr=" + str(learning_rate) + '  ')
                f.write("AUC="+str(AUC)+'\n')
        f.write("MAX_AUC="+str(max_AUC)+'\n')
        MAX_AUC_list.append(max_AUC)
        if index == 4:
            sum = 0
            f.write('(')
            for i, auc in enumerate(MAX_AUC_list):
                sum += auc
                if i < 4:              
                    f.write(str(auc) +'+')
                else:
                    f.write(str(auc))
                    f.write(')/5=')
                    f.write(str(sum/5) + '\n')
