import time
import json
import numpy as np
from gcn.utils import *


def emv(samples, pemv, n=3):
    assert samples.size == pemv.size
    k = float(2/(n+1))
    return samples * k + pemv * (1-k)


def evaluate(sess, model, features, support, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
    outs_val = sess.run([model.outputs_softmax], feed_dict=feed_dict_val)
    return (time.time() - t_test), outs_val[0]


def findNodeEdges(adj):
    nn = adj.shape[0]
    edges = []
    for i in range(nn):
        edges.append(adj.indices[adj.indptr[i]:adj.indptr[i+1]])
    return edges


def isis_v2(edges, nIS_vec_local, cn):
    return np.sum(nIS_vec_local[edges[cn]] == 1) > 0


def isis(edges, nIS_vec_local):
    tmp = (nIS_vec_local==1)
    return np.sum(tmp[edges[0]]*tmp[edges[1]]) > 0


def fake_reduce_graph(adj):
    reduced_node = -np.ones(adj.shape[0])
    reduced_adj = adj
    mapping = np.arange(adj.shape[0])
    reverse_mapping = np.arange(adj.shape[0])
    crt_is_size = 0
    return reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size


def fake_local_search(adj, nIS_vec):
    return nIS_vec.astype(int)


def fairness_metric(s_vec, option='jain', k=2.0):
    cv = lambda x: np.std(x) / np.mean(x)
    if option.lower() == 'jain':
        metric = 1.0/(1.0+cv(s_vec)**2)
    elif option.lower() == 'qoe':
        metric = 1 - 2*np.std(s_vec)/(np.amax(s_vec)-np.amin(s_vec))
    elif option.lower() == 'Gs':
        t1_vec = np.pi * s_vec/(2*np.amax(s_vec))
        t2_vec = np.sin(t1_vec)**(1.0/k)
        metric = np.prod(t2_vec)
    elif option.lower() == 'bossaer':
        t1_vec = s_vec/np.amax(s_vec)
        t2_vec = t1_vec**(1.0/k)
        metric = np.prod(t2_vec)
    else:
        raise NameError('{} fairness metric not supported'.format(option))
    return metric


def ddqn_agent_set(flags):
    """
    Helper function setting the FLAGS of dqn_agent
     --max_degree=1 --predict=mwis --hidden1=32 --num_layer=1 --instances=10 --training_set=IS4SAT
     --feature_size=1 --epsilon_min=0.005 --diver_num=1 --datapath=./data/wireless_train --test_datapath=./data/wireless_test
    """
    flags.FLAGS.training_set = 'IS4SAT'
    flags.FLAGS.predict = 'mwis'
    flags.FLAGS.feature_size = 1
    flags.FLAGS.diver_num = 1
    flags.FLAGS.hidden1 = 32
    flags.FLAGS.num_layer = 1
    flags.FLAGS.epsilon_min = 0.005
    flags.FLAGS.max_degree = 1
    return flags


def cgcn_rss_set(flags):
    """
    Helper function setting the FLAGS of CGCN for rollout search
    --feature_size=32 --epsilon_min=0.005 --diver_num=32 --datapath=./data/wireless_train --test_datapath=./data/wireless_test --max_degree=1 --predict=mwis --hidden1=32 --num_layer=20 --instances=2 --training_set=ERUNI
    """
    flags.FLAGS.training_set = "ERUNI"
    flags.FLAGS.predict = "mwis"
    flags.FLAGS.feature_size = 32
    flags.FLAGS.diver_num = 32
    flags.FLAGS.hidden1 = 32
    flags.FLAGS.num_layer = 20
    flags.FLAGS.epsilon_min = 0.005
    flags.FLAGS.max_degree = 1
    return flags


def write_flags_to_file(filename, FLAGS):
    flags_dict = {}
    for flag_name in FLAGS:
        if not flag_name.startswith('__'):
            flag_value = FLAGS.__flags[flag_name].value
            flags_dict[flag_name] = {'value': flag_value, 'type': type(flag_value).__name__}

    with open(filename, mode='w') as file:
        json.dump(flags_dict, file, indent=4)


def read_flags_from_file(filename):
    config_flags = ConfigFlags()
    with open(filename, mode='r') as file:
        flags_dict = json.load(file)
        for flag_name, flag_data in flags_dict.items():
            flag_value = flag_data['value']
            flag_type = flag_data['type']
            if flag_value:
                setattr(config_flags, flag_name, eval(f'{flag_type}("{flag_value}")'))
    return config_flags


class ConfigFlags:
    pass


def extract_Np(filename):
    list_para = filename[0:-4].split('_')
    N_p = round(float(list_para[2][1:]) * float(list_para[1][1:]), 0)
    return N_p


def extract_N(filename):
    list_para = filename[0:-4].split('_')
    N = int(list_para[1][1:])
    return N


def extract_df_info(df):
    df['N_p'] = df['graph'].apply(extract_Np)
    df['N'] = df['graph'].apply(extract_N)
    return df

