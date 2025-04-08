#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import torch
torch.cuda.set_device(0)
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
from time import time
import shutil
import argparse
import configparser
from model.ASTGNN import make_model
from lib.utils import get_adjacency_matrix, get_adjacency_matrix_2direction, compute_val_loss, predict_and_save_results, load_graphdata_normY_channel1
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from reweighting_rff import lossb_expect
from reweighting_rff import global_local
import random
import datetime
# torch.set_default_dtype(torch.float)



def train_main(alpha, counter,patience,temperature):
    """
    Parameters
    ----------
    counter: early stopping
    patience: threshold of early stoping
    gamma: control the contribution of adjacent matrix
    temperature: temperature

    Returns: best validation loss
    -------
    """

    if  not os.path.exists(params_path):  # 从头开始训练，就要重新构建文件夹
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)   ##delete the direction
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path), flush=True)
    else:
        raise SystemExit('Wrong type of model!')



    criterion_train = nn.L1Loss(reduce = False).to(DEVICE)    ##reduce=False 换成了矩阵形式
    # criterion_train = nn.SmoothL1Loss(reduce=False).to(DEVICE)  ## huber loss square loss in (-1,1)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 优化器可以传入两个参
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
        total_param = total_param +  np.prod(net.state_dict()[param_tensor].size())

    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name], flush=True)
    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    # train model
    if start_epoch > 0:
        # params_filename = os.path.join(params_path, 'temp%s_epoch_%s.params' % (temperature,start_epoch))
        params_filename_pretrain = './experiments/PEMS03/std_o/temp%s_epoch_%s.params' % (temperature, start_epoch)
        print('begin finetuning')
        model_dict = net.state_dict()
        pretrained_dict = torch.load(params_filename_pretrain)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print('start epoch:', start_epoch, flush=True)
        print('load weight from: ', params_filename_pretrain, flush=True)

    start_time = time()
    softmax = nn.Softmax(0)

    pre_features = Variable(torch.zeros(batch_size,num_of_vertices,num_for_predict,d_model).cuda())
    pre_weight_sample1 = Variable(torch.ones(batch_size, 1).cuda())

    weight_sample = Variable(torch.ones(batch_size, 1).cuda())  ## features decorrelation
    weight_sample.requires_grad = True
    weight_node = Variable(torch.ones(num_of_vertices, 1).cuda())  ## features decorrelation
    weight_node.requires_grad = True
    # I = torch.ones(num_of_vertices, num_of_vertices).cuda()
    I = torch.eye(num_of_vertices).cuda()
    weight_nodeNN = Variable(torch.ones(num_of_vertices, num_of_vertices).cuda())  ## features decorrelation
    weight_nodeNN.requires_grad = True
    optimizerbl_n = optim.Adam([weight_nodeNN], lr=learning_rate, eps=1e-08)
      # 定义优化器，传入所有网络参数
    optimizerbl_s = optim.Adam([weight_sample], lr=learning_rate, eps=1e-08)
    weight_T = Variable(torch.ones(12, 1).cuda())  ## features decorrelation
    weight_T.requires_grad = True
    optimizerbl_t = optim.Adam([weight_T], lr=learning_rate, eps=1e-08)
    # temp_scheduler = TemperatureScheduler(optimizer, start_temp=1.0, end_temp=0.1, step_size=10)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler.last_epoch = start_epoch - 1



    # def load_opt_and_weight(path):
    #     state_dicts = torch.load(path, map_location='cuda')
    #     weight_node = state_dicts['weight_node']
    #     weight_sample = state_dicts['weight_sample']
    #     gamma = state_dicts['gamma']
    #     optimizer.load_state_dict(state_dicts['opt_state_dict'])
    #     optimizerbl.load_state_dict(state_dicts['optbl_state_dict'])
    #
    #     return weight_node, weight_sample, gamma, optimizer, optimizerbl

    # if start_epoch > 0:
    #     try:
    #         weight_node, weight_sample, gamma, optimizer, optimizerbl = load_opt_and_weight(os.path.join(params_path, 'epochb_%s.pth' % (start_epoch-1)))
    #         print('load gamma from pth， gamma:', gamma)     ##gamma 没有读进来
    #     except:
    #         print('Error in loading opt and weights.')

    for epoch in range( epochs):

        params_filename = os.path.join(params_path,'temp%s_epoch_%s.params' % (temperature,epoch))
        params_filename_best = os.path.join(params_path, 'temp%s_epoch_%s_best_partition%s.params' % (temperature,epoch,filename))

        # apply model on the validation data set
        # val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch)
        # output = "%d,%d" % (epoch, val_loss)

        net.train()  # ensure dropout layers are in train mode
        for batch_index, batch_data in enumerate(train_loader):
            # decoder_inputs和labels错开一位，decoder_inputs包含t_-1作为decoder起始输入
            encoder_inputs, decoder_inputs, labels = batch_data
            # print(encoder_inputs.shape)  ##18,307,1,12

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            # decoder_inputs1 = decoder_inputs
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
            # print('this is lables1',labels.shape)   ## B,N,T,1
            labels = labels.unsqueeze(-1)

            optimizer.zero_grad()
            # (features_Z,outputs), loss_NCE = net(encoder_inputs, decoder_inputs)
            features_Z, outputs = net(encoder_inputs, decoder_inputs)
            # features_Z.data.copy_(features_Z.data)

            all_feature = torch.cat((features_Z,pre_features.detach()),dim=0)
            if epoch >= 0 :

                for epoch1 in range(0,epochb):
                    # pass

                    # all_weight_sample = torch.cat((weight_sample,pre_weight_sample1.detach()),dim=0)
                    optimizerbl_n.zero_grad()
                    optimizerbl_s.zero_grad()
                    # optimizerbl_t.zero_grad()
                    weight_node = 1 / torch.sum(weight_nodeNN * ((torch.from_numpy(adj_mx).cuda() + alpha * I)), dim=1)
                    weight_node = weight_node.float()
                    weight_sample = weight_sample.float()
                    #
                    loss_reweighting_n = lossb_expect(temperature, adj_mx, features_Z.detach(), softmax(weight_node), \
                                                      num_f=num_of_RFF, sum=True)
                    loss_reweighting_s = lossb_expect(temperature, adj_mx, features_Z.detach(), softmax(weight_sample), \
                                                      num_f=num_of_RFF, sum=True)
                    # # loss_reweighting_t = lossb_expect(temperature, adj_mx, features_Z.detach(), softmax(weight_T), \
                    # #                                   num_f=num_of_RFF, sum=True)
                    loss_reweighting_s.backward(retain_graph=True)
                    optimizerbl_s.step()
                    loss_reweighting_n.backward(retain_graph=True)
                    optimizerbl_n.step()
                    # loss_reweighting_t.backward(retain_graph=True)
                    # optimizerbl_t.step()


            #llm global-local
            pre_features,pre_weight_sample1 = global_local(pre_weight_sample1,pre_features, weight_sample, features_Z, presave_ratio,epoch,batch_index)

            # print('weight_sample',weight_sample,'weight_sample1',weight_sample1)# 1.002 & 0.0083
            ## loss: bxNxTxd ---- NxbxTxd
            # all_weight_sample1 = all_weight_sample1[0:n1, :]
            n1 = features_Z.size(0)
            loss = criterion_train(outputs, labels)  ## n,N,T,d
            # loss1 = masked_mape2(outputs, labels, 0)
            # loss = criterion(outputs, labels)
            # loss = huber(labels, outputs, 0.4)
            # weight_node = torch.sum(weight_nodeNN*(torch.from_numpy(adj_mx).cuda()), dim=1)
            weight_node = 1 / torch.sum(weight_nodeNN * ((torch.from_numpy(adj_mx).cuda() + alpha*I)), dim=1)
            weight_node = weight_node.float()
            loss = torch.matmul(torch.matmul(loss.permute(1, 2, 3, 0), softmax(weight_sample[0:n1, :])).permute(1, 3, 2, 0),
                softmax(weight_node))
            # loss = torch.matmul(loss.permute(1, 2,0), softmax(weight_T))
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step = global_step + 1

        sw.add_scalar('training_loss', training_loss, epoch)

        val_loss = compute_val_loss(net, val_loader, weight_sample,weight_node,criterion_train,sw, epoch)
        # scheduler.step()
        if val_loss <= best_val_loss:
            counter = 0
            best_val_loss = val_loss
            best_epoch = epoch
        else:
            counter += 1
        if counter >= patience and epoch > 40:
            print("early stopping ")
            break
        torch.save(net.state_dict(), params_filename)
        mae, rmse, mape = predict_main(temperature,epoch, test_loader0, test_target_tensor0, _max, _min, 'test')
        mae1, rmse1, mape1 = predict_main1(temperature,epoch, test_loader1, test_target_tensor1, _max1, _min1, 'test1')
        mae2, rmse2, mape2 = predict_main2(temperature,epoch, test_loader2, test_target_tensor2, _max2, _min2, 'test2')
        mae3, rmse3, mape3 = predict_main3(temperature,epoch, test_loader3, test_target_tensor3, _max3, _min3, 'test3')
        output = ("%d,%.4f, " + "mae:" + "%.4f,%.4f,%.4f,%.4f, " + "rmse:" + "%.4f,%.4f,%.4f,%.4f, " + "mape:" + "%.4f,%.4f,%.4f,%.4f") % (
                     epoch, val_loss, mae, mae1, mae2, mae3, rmse, rmse1, rmse2, rmse3, mape, mape1, mape2, mape3)
        with open(loss_txt, 'a+') as f:
            f.write(output + '\n')
            f.close()
    with open(loss_txt, 'a+') as f:
        f.write('best_epoch: %d'% best_epoch + '\n')
        f.close()
    torch.cuda.empty_cache()
    return best_val_loss



    # # fine tune the model
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
    # optimizerbl_n = optim.Adam([weight_nodeNN], lr=learning_rate*0.1)  # 定义优化器，传入所有网络参数
    # optimizerbl_s = optim.Adam([weight_sample], lr=learning_rate * 0.1)
    #
    # for epoch in range(epochs, epochs+fine_tune_epochs):
    #     print('fine tune the model ... ', flush=True)
    #
    #     params_filename = os.path.join(params_path,'temp%s_epoch_%s.params' % (temperature,epoch))
    #     params_filename_best = os.path.join(params_path,'temp%s_epoch_%s_best_partition%s.params' % (temperature,epoch, filename))
    #
    #     net.train()  # ensure dropout layers are in train mode
    #     train_start_time = time()
    #
    #     for batch_index, batch_data in enumerate(train_loader):
    #
    #         encoder_inputs, decoder_inputs, labels = batch_data
    #
    #         encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
    #
    #         decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
    #
    #         labels = labels.unsqueeze(-1)
    #         predict_length = labels.shape[2]  # T
    #
    #         optimizer.zero_grad()
    #         encoder_output = net.encode(encoder_inputs)
    #
    #         # 计算infoNCE_loss
    #         # loss_NCE = net.get_supCR_loss(encoder_inputs,labels,encoder_output)
    #         ## node eange ends
    #
    #         # decode
    #         decoder_start_inputs = decoder_inputs[:, :, :1, :]
    #         decoder_input_list = [decoder_start_innohup python trian  puts]
    #
    #         for step in range(predict_length):
    #             decoder_inputs = torch.cat(decoder_input_list, dim=2)
    #             features_Z,predict_output = net.decode(decoder_inputs, encoder_output)
    #             decoder_input_list = [decoder_start_inputs, predict_output]
    #         all_feature = torch.cat([features_Z, pre_features.detach()], dim=0)
    #
    #
    #         for epoch2 in range(0,epochb):
    #             all_weight_sample = torch.cat((weight_sample, pre_weight_sample1.detach()), dim=0)
    #             optimizerbl_n.zero_grad()
    #             optimizerbl_s.zero_grad()
    #             # weight_node = torch.sum(weight_nodeNN*(torch.from_numpy(adj_mx).cuda()), dim=1)
    #             weight_node = 1 / torch.sum(weight_nodeNN * ((torch.from_numpy(adj_mx).cuda() + I)), dim=1)
    #             loss_reweighting_n = lossb_expect(temperature,adj_mx,features_Z, softmax(weight_node), num_f=num_of_RFF, sum=True)
    #             loss_reweighting_s = lossb_expect(temperature, adj_mx, features_Z, softmax(weight_sample), num_f=num_of_RFF,
    #                                             sum=True)
    #
    #             loss_reweighting_s.backward(retain_graph=True)
    #             optimizerbl_s.step()
    #             loss_reweighting_n.backward(retain_graph=True)
    #             optimizerbl_n.step()
    #
    #
    #             # torch.save(
    #             #     {'epoch': epoch, 'weight_node': weight_node, 'weight_sample': weight_sample,
    #             #      'epochb': epoch2, 'batch_index': batch_index,
    #             #      'opt_state_dict': optimizer.state_dict(), 'optbl_s_state_dict': optimizerbl_s.state_dict()},
    #             #     os.path.join(params_path, 'epoch%s_batch%s_epochb_%s_best.pth' % (epoch, batch_index, epoch1)))
    #
    #             # print("epoch:", str(epoch), ",batch_index:", str(batch_index), ",epoch1:", str(
    #             #     epoch1), ",loss_reweighting_s:", loss_reweighting_s.detach().cpu().numpy())
    #         pre_features,pre_weight_sample1 = global_local(pre_weight_sample1,pre_features, weight_sample, features_Z, presave_ratio,epoch,batch_index)
    #         n1 = features_Z.size(0)
    #         ## loss: bxNxTxd ---- NxTxdxb---TxbxdxN
    #         loss = criterion_train(predict_output, labels)
    #         # loss1 = masked_mape2(predict_output, labels, 0)
    #         # loss = 0.5 * (loss0 + 100 * loss1)
    #         # loss = huber(labels, outputs, 0.4)
    #         # weight_node = torch.sum(weight_nodeNN*(torch.from_numpy(adj_mx).cuda()), dim=1)
    #         # weight_node = 1 / torch.sum(weight_nodeNN * ((torch.from_numpy(adj_mx).cuda() + I)), dim=1)
    #         loss = (torch.matmul(torch.matmul(loss.permute(1, 2, 3, 0), softmax(weight_sample[0:n1, :])).permute(1, 3, 2, 0),
    #             softmax(weight_node)))
    #         loss = torch.mean(loss)
    #         # print("epoch:", epoch, "____loss:", loss, "____loss_NCE:", loss_NCE,  "___tempreture",net.temperature)
    #         # loss = loss+0.02*loss_NCE
    #         loss.backward()
    #         optimizer.step()
    #         training_loss = loss.item()
    #         # global_step = global_step + 1
    #     sw.add_scalar('training_loss', training_loss, epoch)
    #
    #     print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
    #     print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)
    #
    #     # apply model on the validation data set
    #     val_loss = compute_val_loss(net, val_loader, weight_sample,weight_node,criterion_train,sw, epoch)
    #     torch.save(net.state_dict(), params_filename)
    #     if val_loss < best_val_loss:
    #         counter = 0
    #         best_val_loss = val_loss
    #         best_epoch = epoch
    #     else:
    #         counter += 1
    #         torch.save(net.state_dict(), params_filename_best)
    #         # torch.save(
    #         #     {'epoch': epoch, 'weight_node': weight_node, 'weight_sample': weight_sample,
    #         #      'temperature': temperature,
    #         #      'opt_state_dict': optimizer.state_dict(), 'optbl_s_state_dict': optimizerbl_s.state_dict()},
    #         #     os.path.join(params_path, 'temp%s_epochb_%s_best.pth' % (temperature,epoch)))
    #
    #     if counter >= patience and epoch >= 40:
    #         print("early stopping after 40 epoch")
    #         break
    #
    #     mae, rmse, mape = predict_main(temperature, epoch, test_loader0, test_target_tensor0, _max, _min, 'test')
    #     mae1, rmse1, mape1 = predict_main1(temperature, epoch, test_loader1, test_target_tensor1, _max, _min, 'test1')
    #     mae2, rmse2, mape2 = predict_main2(temperature, epoch, test_loader2, test_target_tensor2, _max, _min, 'test2')
    #     mae3, rmse3, mape3 = predict_main3(temperature,epoch, test_loader3, test_target_tensor3, _max, _min, 'test3')
    #     output = ("%d,%.4f, " + "mae:" + "%.4f,%.4f,%.4f,%.4f, " + "rmse:" + "%.4f,%.4f,%.4f,%.4f, " + "mape:" + "%.4f,%.4f,%.4f,%.4f") % (
    #     epoch, val_loss, mae, mae1, mae2, mae3, rmse, rmse1, rmse2, rmse3, mape, mape1, mape2, mape3)
    #     with open(loss_txt, 'a+') as f:
    #         f.write(output + '\n')
    #         f.close()
    # with open(loss_txt, 'a+') as f:
    #     f.write('best_epoch: %d' % best_epoch + '\n')
    #     f.close()
    # print('best epoch:', best_epoch, flush=True)
    # print('apply the best val model on the test data set ...', flush=True)
    # return best_val_loss

def predict_main(temperature, epoch, data_loader, data_target_tensor, _max, _min, type):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''
    params_filename = os.path.join(params_path, 'temp%s_epoch_%s.params' % (temperature,epoch))
    net.load_state_dict(torch.load(params_filename))
    mae, rmse, mape = predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path,type)
    return mae, rmse, mape
def predict_main1(temperature,epoch, data_loader, data_target_tensor, _max, _min, type):
    params_filename = os.path.join(params_path, 'temp%s_epoch_%s.params' % (temperature,epoch))
    net1.load_state_dict(torch.load(params_filename))
    mae, rmse, mape = predict_and_save_results(net1, data_loader, data_target_tensor, epoch, _max, _min, params_path,type)
    return mae, rmse, mape
def predict_main2(temperature,epoch, data_loader, data_target_tensor, _max, _min, type):
    params_filename = os.path.join(params_path,'temp%s_epoch_%s.params' % (temperature,epoch))
    net2.load_state_dict(torch.load(params_filename))
    mae, rmse, mape = predict_and_save_results(net2, data_loader, data_target_tensor, epoch, _max, _min, params_path,type)
    return mae, rmse, mape
def predict_main3(temperature, epoch, data_loader, data_target_tensor, _max, _min, type):
    params_filename = os.path.join(params_path, 'temp%s_epoch_%s.params' % (temperature,epoch))
    net3.load_state_dict(torch.load(params_filename))
    mae, rmse, mape = predict_and_save_results(net3, data_loader, data_target_tensor, epoch, _max, _min, params_path,type)
    return mae, rmse, mape

if __name__ == "__main__":
    # print("pid",os.getpid())
    pattern = 'ns6'
    filename = 'b144i30'+pattern+'_03_100'
    num_of_RFF = 1
    alpha = 6
    loss_txt = 'results/loss_all/' \
               '%s.txt' % (filename)
    #grid research
    best_score = 0
    for temperature in [2.0]:
        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
        seed1 = 2000
        setup_seed(seed1)    ## original is 20

        # read hyper-param settings
        parser = argparse.ArgumentParser()
        # parser.add_argument("--config", default='../traffic-speed/metr-LA/part0/METR-LA.conf', type=str,
        #                     help="configuration file path")
        # parser.add_argument("--config", default='../traffic-flow/PEMSD3-Stream/partition/PEMSD3-Stream.conf', type=str,
        #                     help="configuration file path")
        parser.add_argument("--config", default='data/PEMS03/data_partitions_PEMS03/part0/PEMS03_all.conf', type=str,
                            help="configuration file path")
        parser.add_argument('--cuda', type=str, default='0')
        args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # USE_CUDA = "0"
        DEVICE = torch.device('cuda:0')

        print("CUDA:",  DEVICE, flush=True)

        config = configparser.ConfigParser()
        print('Read configuration file: %s' % (args.config), flush=True)
        config.read(args.config)
        data_config = config['Data']
        training_config = config['Training']
        adj_filename = data_config['adj_filename']
        adj_filename1 = data_config['adj_filename1']
        adj_filename2 = data_config['adj_filename2']
        adj_filename3 = data_config['adj_filename3']
        graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
        graph_signal_matrix_filename_test1 = data_config['graph_signal_matrix_filename_test1']
        graph_signal_matrix_filename_test2 = data_config['graph_signal_matrix_filename_test2']
        graph_signal_matrix_filename_test3 = data_config['graph_signal_matrix_filename_test3']
        if config.has_option('Data', 'id_filename'):
            id_filename = data_config['id_filename']
        else:
            id_filename = None
        if config.has_option('Data', 'id_filename1'):
            id_filename1 = data_config['id_filename1']
        else:
            id_filename1 = None
        if config.has_option('Data', 'id_filename2'):
            id_filename2 = data_config['id_filename2']
        else:
            id_filename2 = None
        if config.has_option('Data', 'id_filename3'):
            id_filename3 = data_config['id_filename3']
        else:
            id_filename3 = None
        num_of_vertices = int(data_config['num_of_vertices'])
        points_per_hour = int(data_config['points_per_hour'])
        num_for_predict = int(data_config['num_for_predict'])
        dataset_name = data_config['dataset_name']
        model_name = training_config['model_name']
        learning_rate = float(training_config['learning_rate'])
        start_epoch = int(training_config['start_epoch'])
        epochs = int(training_config['epochs'])
        epochb = int(training_config['epochb'])
        fine_tune_epochs = int(training_config['fine_tune_epochs'])
        print('total training epoch, fine tune epoch:', epochs, ',', fine_tune_epochs, flush=True)
        batch_size = int(training_config['batch_size'])
        print('batch_size:', batch_size, flush=True)
        num_of_weeks = int(training_config['num_of_weeks'])
        num_of_days = int(training_config['num_of_days'])
        num_of_hours = int(training_config['num_of_hours'])
        direction = int(training_config['direction'])
        encoder_input_size = int(training_config['encoder_input_size'])
        decoder_input_size = int(training_config['decoder_input_size'])
        dropout = float(training_config['dropout'])
        kernel_size = int(training_config['kernel_size'])
        presave_ratio = float(training_config['presave_ratio'])

        # filename_npz = os.path.join(
        #     dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_demo.npz'
        num_layers = int(training_config['num_layers'])
        d_model = int(training_config['d_model'])
        nb_head = int(training_config['nb_head'])
        ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
        SE = bool(int(training_config['SE']))  # whether use spatial embedding
        smooth_layer_num = int(training_config['smooth_layer_num'])
        aware_temporal_context = bool(int(training_config['aware_temporal_context']))
        TE = bool(int(training_config['TE']))
        use_LayerNorm = True
        residual_connection = True

        # direction = 1 means: if i connected to j, adj[i,j]=1;
        # direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
        if direction == 2:
            adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
            adj_mx1, distance_mx1 = get_adjacency_matrix_2direction(adj_filename1, num_of_vertices, id_filename1)
            adj_mx2, distance_mx2 = get_adjacency_matrix_2direction(adj_filename2, num_of_vertices, id_filename2)
            adj_mx3, distance_mx3 = get_adjacency_matrix_2direction(adj_filename3, num_of_vertices, id_filename3)
            # META   LLM
            # adj_mx = pd.read_csv(adj_filename,header=None).values
            # adj_mx1 = pd.read_csv(adj_filename1,header=None).values
            # adj_mx2 = pd.read_csv(adj_filename2,header=None).values
            # adj_mx3 = pd.read_csv(adj_filename3,header=None).values
            # adj_mx[adj_mx > 1e-8] = 1
            # # print(np.sum(adj_mx>1e-4))
            # adj_mx1[adj_mx1 > 1e-8] = 1
            # adj_mx2[adj_mx2 > 1e-8] = 1
            # adj_mx2[adj_mx2 > 1e-8] = 1
            # adj_mx = adj_mx.astype(np.float32)
            # adj_mx1 = adj_mx1.astype(np.float32)
            # adj_mx2 = adj_mx2.astype(np.float32)
            # adj_mx3 = adj_mx3.astype(np.float32)
        if direction == 1:
            adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
            adj_mx1, distance_mx1 = get_adjacency_matrix(adj_filename1, num_of_vertices, id_filename1)
            adj_mx2, distance_mx2 = get_adjacency_matrix(adj_filename2, num_of_vertices, id_filename2)
            adj_mx3, distance_mx3 = get_adjacency_matrix(adj_filename3, num_of_vertices, id_filename3)

        folder_dir = 'std_%s_MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (pattern,
        model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, encoder_input_size,
        direction, dropout, learning_rate)

        if aware_temporal_context:
            folder_dir = folder_dir + 'Tcontext'
        if ScaledSAt:
            folder_dir = folder_dir + 'ScaledSAt'
        if SE:
            folder_dir = folder_dir + 'SE' + str(smooth_layer_num)
        if TE:
            folder_dir = folder_dir + 'TE'

        print('folder_dir:', folder_dir, flush=True)
        params_path = os.path.join('./experiments', dataset_name, folder_dir)
        print("params_path", params_path)
        # all the input has been normalized into range [-1,1] by MaxMin normalization
        train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader0, test_target_tensor0, test_loader1, test_target_tensor1, test_loader2, test_target_tensor2, test_loader3, test_target_tensor3, _max, _min,_max1,_min1,_max2,_min2,_max3,_min3 = load_graphdata_normY_channel1(
            graph_signal_matrix_filename, graph_signal_matrix_filename_test1, graph_signal_matrix_filename_test2,
            graph_signal_matrix_filename_test3, num_of_hours,
            num_of_days, num_of_weeks, DEVICE, batch_size)

        # train_main()
        # adj_mx = np.zeros((int(num_of_vertices), int(num_of_vertices)),
        #                   dtype=np.float32)
        # adj_mx1 = np.zeros((int(num_of_vertices), int(num_of_vertices)),
        #                   dtype=np.float32)
        # adj_mx2 = np.zeros((int(num_of_vertices), int(num_of_vertices)),
        #                   dtype=np.float32)
        # adj_mx3 = np.zeros((int(num_of_vertices), int(num_of_vertices)),
        #                   dtype=np.float32)

        net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx,
                         nb_head,
                         num_of_weeks,
                         num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout,
                         aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE, TE=TE,
                         kernel_size=kernel_size, smooth_layer_num=smooth_layer_num,
                         residual_connection=residual_connection,
                         use_LayerNorm=use_LayerNorm)
        net1 = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx1,
                          nb_head,
                          num_of_weeks,
                          num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout,
                          aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE, TE=TE,
                          kernel_size=kernel_size, smooth_layer_num=smooth_layer_num,
                          residual_connection=residual_connection,
                          use_LayerNorm=use_LayerNorm)

        net2 = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx2,
                          nb_head,
                          num_of_weeks,
                          num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout,
                          aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE, TE=TE,
                          kernel_size=kernel_size, smooth_layer_num=smooth_layer_num,
                          residual_connection=residual_connection,
                          use_LayerNorm=use_LayerNorm)
        net3 = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx3,
                          nb_head,
                          num_of_weeks,
                          num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout,
                          aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE, TE=TE,
                          kernel_size=kernel_size, smooth_layer_num=smooth_layer_num,
                          residual_connection=residual_connection,
                          use_LayerNorm=use_LayerNorm)
        counter0 = 0
        patience0 = 10
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        with open(loss_txt, 'a+') as f:
            f.write(' z-score , %s bay dim=3+2  I=eye  3090ti  alpha=%   seed=%s all data'%(pattern,alpha,seed1) + '\n')
            f.write(formatted_datetime + '\n')
            f.close()
        score = train_main(alpha, counter0,patience0,temperature)
        if score > best_score:
            best_score = score
            best_parameters = {'temperature': temperature}
    #   grid search end
    print("Best score:{:.2f}".format(best_score))
    print("Best temperature:{}".format(best_parameters))

    # train_main()


    # predict_main(0, test_loader, test_target_tensor, _max, _min, 'test')
















