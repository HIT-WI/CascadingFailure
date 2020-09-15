import numpy as np
import torch
import networkx as nx
from scipy.special import comb
import math
import collections

DG = nx.DiGraph()
SAMPLE_NUM = 0
MAX_DEGREE_E = 0
MAX_DEGREE_D = 0
node_examine_num = 0
node_disease_num = 0
degree_avg_E = 0
degree_avg_D = 0

TIMESTEPS = 120

def read_data_init(filePath_init, filePath_death):
    global SAMPLE_NUM

    data_input = []
    fail_node_all = []
    with open(filePath_init, "r") as file_init:
        for line in file_init.readlines():
            failure_pro_init = float(str(line.strip('\n')).split('\t')[1])
            data_input.append(1 - failure_pro_init)
            fail_node_all.append(line.strip('\n').split('\t')[2].split('::'))

    data_death = []
    with open(filePath_death, "r") as file_death:
        for line in file_death.readlines():
            is_death = float(str(line.strip('\n')).split('\t')[1])
            data_death.append(is_death)

    SAMPLE_NUM = len(data_input)
    return data_input, data_death, fail_node_all


def build_data_graph(filePath_weight, filePath_graph):
    global MAX_DEGREE_E
    global MAX_DEGREE_D
    global node_examine_num
    global node_disease_num
    global degree_avg_E
    global degree_avg_D

    with open(filePath_graph, "r") as file_graph:
        for line in file_graph.readlines():
            elements = str(line.strip('\n')).split('\t')

            if not DG.__contains__(elements[0]):
                DG.add_node(elements[0], type='examine')
            if not DG.__contains__(elements[1]):
                DG.add_node(elements[1], type='disease')

    with open(filePath_weight, "r") as file_weight:
        for line in file_weight.readlines():
            # print(line)
            elements = str(line.strip('\n')).split('::')[0].split('-->')
            weight = float(str(line.strip('\n')).split('::')[1])

            if not DG.has_edge(elements[0], elements[1]):
                DG.add_edge(elements[0], elements[1], weight=weight)

    degree_dict_e = {}
    degree_dict_d = {}

    for node in DG.nodes(data=True):
        if node[1]['type'] is 'examine':
            node_examine_num += 1
            degree = DG.in_degree(node[0])
            degree_avg_E += degree
            if degree in degree_dict_e:
                degree_dict_e[degree].append(node[0])
            else:
                degree_dict_e.setdefault(degree, []).append(node[0])
        else:
            node_disease_num += 1
            degree = DG.in_degree(node[0])
            degree_avg_D += degree
            if degree in degree_dict_d:
                degree_dict_d[degree].append(node[0])
            else:
                degree_dict_d.setdefault(degree, []).append(node[0])

    degree_avg_E /= node_examine_num
    degree_avg_D /= node_disease_num

    degree_dict_e_sorted = sorted(degree_dict_e.items(), key=lambda x: x[0])
    degree_dict_d_sorted = sorted(degree_dict_d.items(), key=lambda x: x[0])
    MAX_DEGREE_E = degree_dict_e_sorted[-1][0]
    MAX_DEGREE_D = degree_dict_d_sorted[-1][0]

    return degree_dict_e_sorted, degree_dict_d_sorted


def generate_r(degree_dic_sorted):
    flag = 1
    for degree_node in degree_dic_sorted:
        for node in degree_node[1]:
            node_j = np.zeros((degree_node[0] + 1, 1))
            neighbor_alpha = []
            for neighbor in DG.neighbors(node):
                neighbor_alpha.append(DG[neighbor][node]['weight'])
                neighbor_alpha = sorted(neighbor_alpha, reverse=True)

            for j in range(degree_node[0] + 1):
                if j == 0:
                    continue
                else:
                    node_j[j, 0] = (sum(neighbor_alpha[:j]) + sum(neighbor_alpha[-j:])) / 2
            if flag == 1:
                sample = node_j
                flag = 0
            else:
                sample = np.concatenate((sample, node_j), axis=0)

    return torch.tensor(sample)


def cal_function_W(word_fail, label, x_input, beta, network_type):
    W = 0
    node_ep = []
    node_effetive_probability = {}
    fail_node_exam = []
    for degree, sample in x_input:
        total_temp = 0
        for node in sample:
            effetive_probability = 0
            if label[node] == "0":
                if node in word_fail:
                    m = len(word_fail[node])
                else:
                    m = 0
                for j in range(degree + 1 - m):
                    if network_type:
                        index = node_index_dic_e[node] + j
                        r = r_e[index].squeeze()
                    else:
                        index = node_index_dic_d[node] + j
                        r = r_d[index].squeeze()

                    # r = 3 * math.pow(j / degree, 2) - 2 * math.pow(j / degree, 3)

                    effetive_probability += comb(degree - m, j) * math.pow(beta, j) * math.pow(1 - beta,
                                                                                                   degree - m - j) * r
            if node not in node_effetive_probability:
                node_effetive_probability[node] = effetive_probability
            node_ep.append(effetive_probability)
            total_temp += effetive_probability

        if network_type:
            W += total_temp / node_examine_num
        else:
            W += total_temp / node_disease_num

    node_effetive_probability_sorted = sorted(node_effetive_probability, key=node_effetive_probability.__getitem__,
                                              reverse=False)
    if network_type:
        fail_node_num = node_examine_num - int(W * node_examine_num)
        if fail_node_exam_all:
            total = fail_node_num - len(fail_node_exam_all[-1])
        else:
            total = fail_node_num
        flag = 0
        for node in node_effetive_probability_sorted:
            if flag >= total: break
            if fail_node_exam_all:
                if node not in fail_node_exam_all[-1]:
                    fail_node_exam.append(node)
                    flag += 1
            else:
                fail_node_exam.append(node)
                flag += 1

        if fail_node_exam_all:
            fail_node_exam_all.append(fail_node_exam_all[-1] + fail_node_exam)
        else:
            fail_node_exam_all.append(fail_node_exam)

    return W, node_ep


def cal_function_Z(word_fail, label, x_input, beta, network_type):
    Z = 0

    for degree, sample in x_input:
        total_temp = 0
        for node in sample:
            effetive_probability = 0
            if label[node] == "0":
                if node in word_fail:
                    m = len(word_fail[node])
                else:
                    m = 0

                for j in range(degree - m):
                    if network_type:
                        index = node_index_dic_e[node] + j + 1
                        r = r_e[index]
                    else:
                        index = node_index_dic_d[node] + j + 1
                        r = r_d[index]
                    # r = 3 * math.pow((j + 1) / degree, 2) - 2 * math.pow((j + 1) / degree, 3)

                    total_temp += comb(degree - 1 - m, j) * math.pow(beta, j) * math.pow(1 - beta,
                                                                                         degree - 1 - m - j) * r
            total_temp += effetive_probability
        if network_type:
            Z += degree * total_temp / (degree_avg_E * node_examine_num)
        else:
            Z += degree * total_temp / (degree_avg_D * node_disease_num)

    return Z


department_id = ['7', '12', '15', '23', '50', '52']
for id in department_id:
    data_input, data_death, fail_node_all = read_data_init('./data/' + id + '/Initial_failure_probability_20_' + id + '_dis.txt',
                                            './data/' + id + '/hadm_death_20_' + id + '.txt'
                                            )

    degree_dict_e_sorted, degree_dict_d_sorted = build_data_graph('./data/' + id + '/weight_' + id + '.txt',
                                                                  './data/' + id + '/exam_disease_20-' + id + '.txt')

    print('SAMPLE_NUM::', SAMPLE_NUM, ';MAX_DEGREE_E::', MAX_DEGREE_E, ';MAX_DEGREE_D::', MAX_DEGREE_D,
          ';node_examine_num::', node_examine_num, ';node_disease_num::', node_disease_num, ';degree_avg_E::', degree_avg_E,
          ';degree_avg_D::', degree_avg_D)

    r_e = generate_r(degree_dict_e_sorted)
    r_d = generate_r(degree_dict_d_sorted)

    node_index_dic_e = collections.OrderedDict()
    node_index_dic_d = collections.OrderedDict()
    index = 0
    for degree, nodes in degree_dict_e_sorted:
        for node in nodes:
            node_index_dic_e[node] = index
            index += degree + 1
    index = 0
    for degree, nodes in degree_dict_d_sorted:
        for node in nodes:
            node_index_dic_d[node] = index
            index += degree + 1

    target_exam = []
    with open("./data/" + id + "/hadm_record_20_" + id + ".txt", "r") as f:
        for row in f.readlines():
            r = row.strip().split('\t')
            target_exam.append(r[1].split('::')[0].split(' '))

    effetive_probability = []
    E_miu_all = []
    D_miu_all = []
    recall_all = []
    for sample_index in range(SAMPLE_NUM):
        word_label = {}
        word_fail_dic = {}
        fail_node = []
        fail_node_exam_all = []

        for node in DG.nodes:
            if node in fail_node_all[sample_index]:
                word_label[node] = "1"
            else:
                word_label[node] = "0"

        for node in fail_node_all[sample_index]:
            for neighbor in DG.neighbors(node):
                if neighbor not in word_fail_dic:
                    word_fail_dic[neighbor] = []
                if node not in word_fail_dic[neighbor]:
                    word_fail_dic[neighbor].append(node)

        miu = np.zeros(TIMESTEPS + 1)
        miu[0] = data_input[sample_index]
        state = np.zeros(TIMESTEPS + 1)
        state[0] = data_input[sample_index]

        node_effetive_probability_e = []
        node_effetive_probability_d = []
        for time_step in range(TIMESTEPS):
            if time_step % 2 == 0:
                miu_now, node_ep = cal_function_W(word_fail_dic, word_label, degree_dict_e_sorted, state[time_step], True)
                miu[time_step + 1] = miu_now
                node_effetive_probability_e.append(node_ep)
                h_state_temp = cal_function_Z(word_fail_dic, word_label, degree_dict_e_sorted, state[time_step], True)
                state[time_step + 1] = h_state_temp
            else:
                miu_now, node_ep = cal_function_W(word_fail_dic, word_label, degree_dict_d_sorted, state[time_step], False)
                miu[time_step + 1] = state[0] * miu_now
                node_effetive_probability_d.append(node_ep)
                h_state_temp = cal_function_Z(word_fail_dic, word_label, degree_dict_d_sorted, state[time_step], False)
                state[time_step + 1] = state[0] * h_state_temp

        node_effetive_probability = np.concatenate((np.array(node_effetive_probability_e), np.array(node_effetive_probability_d)), axis=1)
        effetive_probability.append(node_effetive_probability)

        recall = []
        for fail_node in fail_node_exam_all:
            num = 0
            for node in fail_node:
                if node in target_exam[sample_index]:
                    num += 1
            recall.append(num / len(target_exam[sample_index]))

        D_miu = miu[::2]
        E_miu = miu[1::2]
        E_miu = list(E_miu)
        E_miu.insert(0, 1)
        recall.insert(0, 0)

        E_miu_all.append(E_miu)
        D_miu_all.append(D_miu)
        recall_all.append(recall)

    effetive_pro = np.array(effetive_probability)

    path_name = './data/' + id + '/CF_node_eff_pro.npy'
    np.save(path_name, effetive_pro)
    np.save('./data/' + id + '/E_miu_' + id + '.npy', np.array(E_miu_all))
    np.save('./data/' + id + '/D_miu_' + id + '.npy', np.array(D_miu_all))
    np.save('./data/' + id + '/recall_' + id + '.npy', np.array(recall_all))

    # path_name = './data/' + id + '/CF_node_polynomial_eff_pro.npy'
    # np.save(path_name, effetive_pro)
    # np.save('./data/' + id + '/polynomial_E_miu_' + id + '.npy', np.array(E_miu_all))
    # np.save('./data/' + id + '/polynomial_D_miu_' + id + '.npy', np.array(D_miu_all))
    # np.save('./data/' + id + '/polynomial_recall_' + id + '.npy', np.array(recall_all))
