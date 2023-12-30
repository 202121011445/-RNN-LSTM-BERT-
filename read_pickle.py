import os.path
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import torch as d2l


if __name__ == "__main__":
    file = open('state_dict/log.pickle', 'rb+')
    data = pickle.load(file)
    plt.plot(data['train_auc'], ls='--', c='r', label='train_auc')
    plt.plot(data['test_auc'], c='r', label='test_auc')
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.grid()
    plt.title("BERT")
    plt.legend()
    plt.show()

    # hiddens = [512]
    # layers = [3]
    # epochs = [0]
    # test_auc = {}
    # path_model = 'lstm hiddens{0} layers{1} epoch{2}.pickle'
    # for hidden in hiddens:
    #     for layer in layers:
    #         for epoch in epochs:
    #             log_path = path_model.format(hidden, layer, epoch)
    #             if os.path.isfile(log_path):
    #                 file = open(log_path, 'rb+')
    #                 data = pickle.load(file)
    #                 test_auc[log_path] = max(data['test_auc'])
    # print(sorted(test_auc.items(), key=lambda x: x[1], reverse=True))
    # test_auc = sorted(test_auc.items(), key=lambda x: x[1], reverse=True)[0]
    # file = open(test_auc[0], 'rb+')
    # data = pickle.load(file)
    # print('最大的测试集AUC出现为{0}，出现在{1}中.'.format(max(data['test_auc']),test_auc[0]))
    # plt.plot(data['train_auc'], ls='--', c='r', label='train_auc')
    # plt.plot(data['test_auc'], c='r', label='test_auc')
    # plt.xlabel('epoch')
    # plt.ylabel('auc')
    # plt.grid()
    # plt.legend()
    # plt.show()


    # file = open('lstm hiddens512 layers3 epoch0.pickle', 'rb+')
    # # file = open('rnn hiddens128 layers2 epoch1.pickle', 'rb+')
    # data = pickle.load(file)
    # plt.plot(data['train_auc'], ls='--', c='r', label='train_auc')
    # plt.plot(data['test_auc'], c='r', label='test_auc')
    # plt.xlabel('epoch')
    # plt.ylabel('auc')
    # plt.title("rnn")
    # plt.grid()
    # plt.legend()
    # plt.show()
