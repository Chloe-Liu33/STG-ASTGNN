import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from lib.utils import norm_Adj
from torch.autograd import Variable
softmax = nn.Softmax(0)
# torch.set_default_dtype(torch.float)


## llm
def RFF(feature, w=None, sigma=None, num_f=None, sum=True):
    if num_f is None:
        num_f = 1
    n = feature.size(0)
    r = feature.size(1)
    c = feature.size(2)
    d = feature.size(3)
    feature = feature.view(n, r, c, d, 1)
    e = feature.size(4)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:     # w is weight for the triangle function
        ##w,b随机初始化
        # w = 1 / sigma * (torch.randn(size=(num_f, c)))
        w = 1 / sigma * (torch.randn(size=(num_f, e)))
        # w = w.unsqueeze(dim=0)   ##increase one dimension
        # print('w:',w.shape)  ##1x1
        #  this is random for N,T,d
        # b = 2 * np.pi * torch.rand(size=(r,c,d, num_f))
        # this is random for d only
        b = 2 * np.pi * torch.rand(size=( d, num_f))
        # b = b.repeat((n,1,1,1,1))    #bxNxTxdx1      ##llm重新调整
        b = b.repeat(n,r,c,1,1)
        # print('b:',b)  ##  normal value
        # print('features:',feature.shape)

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())
    # w = torch.tensor(w)  ##llm自己加的
    # w = w.clone().detach().requires_grad_(True)
    mid = torch.matmul(feature.cuda(), w.t().cuda())  ## 'int' object has no attribute 't'
    mid = mid + b.cuda()
    # mid_ori_shape = mid.shape
    # mid_max = mid.reshape(mid_ori_shape[0], -1).max(dim=1, keepdim=True)[0].view(mid_ori_shape[0], 1, 1, 1, 1)
    # mid_min = mid.reshape(mid_ori_shape[0], -1).min(dim=1, keepdim=True)[0].view(mid_ori_shape[0], 1, 1, 1, 1)
    # # mid_max = mid.max(dim=3, keepdim=True)[0].cuda()
    # # mid_max = mid_max.max(dim=2, keepdim=True)[0].cuda()
    # # mid_max = mid_max.max(dim=1, keepdim=True)[0].cuda()
    # # mid_min = mid.min(dim=3, keepdim=True)[0].cuda()
    # mid = (mid - mid_min) / (mid_max - mid_min)
    # mid1 = mid
    # mid1 -= mid1.min(dim=1, keepdim=True)[0]
    # mid1 /= mid1.max(dim=1, keepdim=True)[0]
    mid3 = mid
    mid3 -= mid3.min(dim=3, keepdim=True)[0].min(dim=2,keepdim=True)[0]      ###数据的归一化将从两个不同的维度进行
    mid3 /= mid3.max(dim=3, keepdim=True)[0].max(dim=2,keepdim=True)[0]
    # mid3 -= mid3.min(dim=3, keepdim=True)[0].min(dim=2,keepdim=True)[0]
    # mid3 /= mid3.max(dim=3, keepdim=True)[0].max(dim=2,keepdim=True)[0]
    if torch.isnan(mid3).any():
        print('Nan detected for mid')
    # print('mid', mid)   ## normal value and nan
    # mid1 *= np.pi / 2.0     #n,r,1
    mid3 *= np.pi / 2.0

    if sum:
        # Z1 = Z * (torch.cos(mid1).cuda() + torch.sin(mid1).cuda())
        Z3 = Z * (torch.cos(mid3).cuda() + torch.sin(mid3).cuda())
    else:
        # Z1 = Z * torch.cat((torch.cos(mid1).cuda(), torch.sin(mid1).cuda()), dim=-1)
        Z3 = Z * torch.cat((torch.cos(mid3).cuda(), torch.sin(mid3).cuda()), dim=-1)

    return Z3

def cov_sample(x, w=None):   ##covariance
    # num = x.size(0)
    if w == None:
        # w1 = w[0:num,:]
        # w1 = w1.view(-1,1)
        # cov = torch.matmul((w1 * x).t(), x)
        # e = torch.sum(w1 * x, dim=0).view(-1, 1)
        # res = cov - torch.matmul(e, e.t())
        ##这里没有权重，计算整个矩阵的协方差
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        w = w[0:x.size(0), :]
        # w = w[0:x.size(0)*100,:]

        #x:bxNxTxd    cov:NxTxdxd
        # print('weight:',w.shape,'x:',x.shape)  ## bx1, bxNxTxd  * means the last two dimensions multiplicaiton
        x= x.permute(1,2,0,3)  ## NxTxbxd
        cov = torch.matmul((w * x).permute(0,1,3,2), x)   ##  N,T,d,d,,第二个size

        e = torch.unsqueeze(torch.sum(w * x, dim=2),2).permute(0,1,3,2)   #(100,12,d,1)   dim=2  is n dimension
        res = cov - torch.matmul(e, e.transpose(3,2))    ##2D matrix 高维度矩阵采用转置只交换最后两列

        ##Nxs
        # a1 = x.size(2)
        # b1 = x.size(3)
        # x = x.view(-1,a1,b1)   ##bx100,12,64
        # # print('x',x.shape,'w',w.shape)
        # x1= x.permute(1,0,2)  ## 12,Nxs,64
        # cov = torch.matmul((w * x1).permute(0,2,1), x1) ##  12,64,64
        # e = torch.sum(w * x1, dim=1).reshape(12,1,64)##12,1,64
        # # print('e',e.shape)
        # res = cov - torch.matmul(e.transpose(2,1),e)    ##2D matrix 高维度矩阵采用转置只交换最后两列

        ##squeezze the n and T dimension(the  new calculation method)
        # x = torch.mean(x, axis=1)  ##nxTxd
        # x = torch.mean(x, axis=1)  ##nxd
        # cov = torch.matmul((w * x).t(), x)
        # e = torch.sum((w * x), axis=0).view(-1, 1)
        # res = cov - torch.matmul(e, e.t())
        #计算交叉协方差
    return res


# laplacian矩阵
def unnormalized_laplacian(adj_matrix):
    # 先求度矩阵
    R = np.sum(adj_matrix, axis=1)
    degreeMatrix = np.diag(R)
    return degreeMatrix - adj_matrix

# 随机漫步归一化邻接矩阵
def normalized_laplacian2(gamma, adj_matrix):
    # print('adjmatrix',adj_matrix+torch.float(torch.eye(adj_matrix.size(0))))
    # print('adjmatrix', adj_matrix)
    R = np.sum(adj_matrix, axis=1)
    # print("R",R)   #r 出现了0
    R = np.where(R==0,1000000,R)   ####R 应该取哪个值？？？
    R_frac = 1 / R
    D_frac = np.diag(R_frac)
    # D_sqrt = D_sqrt.fillna(0)   ## this is for data frame
    # D_sqrt[np.isnan(D_sqrt)]=0   ## this is for numpy
    # print(D_sqrt)
    I = np.eye(adj_matrix.shape[0])
    I = torch.tensor(I, device='cuda')
    return I + gamma*torch.tensor(np.matmul(D_frac, adj_matrix),device="cuda")
## 对称归一化邻接矩阵
def normalized_laplacian3(gamma, adj_matrix):
    # print('adjmatrix',adj_matrix+torch.float(torch.eye(adj_matrix.size(0))))
    # print('adjmatrix', adj_matrix)
    R = np.sum(adj_matrix, axis=1)
    # print("R",R)   #r 出现了0
    R = np.where(R==0,1000000,R)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    # D_sqrt = D_sqrt.fillna(0)   ## this is for data frame
    # D_sqrt[np.isnan(D_sqrt)]=0   ## this is for numpy
    # print(D_sqrt)
    I = np.eye(adj_matrix.shape[0])
    I = torch.tensor(I, device='cuda')
    # print('this is lapalas')
    # print(I+gamma*np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt))
    return I+gamma*torch.tensor(np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt),devide="cuda")

def cov_node(gamma, adj_matrix0,x, w=None):  ##covariance
    # num = x.size(0)
    if w == None:
        # w1 = w[0:num,:]
        # w1 = w1.view(-1,1)
        # cov = torch.matmul((w1 * x).t(), x)
        # e = torch.sum(w1 * x, dim=0).view(-1, 1)
        # res = cov - torch.matmul(e, e.t())
        ##这里没有权重，计算整个矩阵的协方差
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        ##x:bxNxTxd
        x = x.permute(0,2,1,3)   ##bxTxNxd

        cov = torch.matmul((w * x).permute(0, 1, 3, 2), x)
        # cov = torch.matmul(torch.matmul((w * x).permute(0, 1, 3, 2),adj_matrix), x)
        e = torch.unsqueeze(torch.sum(w * x, dim=2),2).permute(0, 1, 3, 2)
        res = cov - torch.matmul(e, e.transpose(3,2))   ##cross-entropy calculation // 2n,12,64,64

    return res


def cov_T(x, w=None):  ##covariance
    # num = x.size(0)
    if w == None:

        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        ##x:bxNxTxd
        cov = torch.matmul((w * x).permute(0, 1, 3, 2), x)
        # cov = torch.matmul(torch.matmul((w * x).permute(0, 1, 3, 2),adj_matrix), x)
        e = torch.unsqueeze(torch.sum(w * x, dim=2),2).permute(0, 1, 3, 2)
        res = cov - torch.matmul(e, e.transpose(3,2))   ##cross-entropy calculation // 2n,12,64,64
    return res
##llm
def lossb_expect(gamma, adj_matrix,cfeaturec, weight, num_f, sum):
    if num_f == 0:
        n = cfeaturec.size(0)
        r = cfeaturec.size(1)
        c = cfeaturec.size(2)
        d = cfeaturec.size(3)
        cfeaturec3 = cfeaturec.view(n, r, c, d, 1)
    else:
        cfeaturec3  = RFF(cfeaturec,num_f=num_f, sum=sum)
        cfeaturec3 = cfeaturec3.cuda()
        # if torch.isnan(cfeaturecs).any():
        #     RFF(cfeaturec, num_f=num_f, sum=sum).cuda()

    loss = Variable(torch.FloatTensor([0]).cuda())

    for i in range(cfeaturec3.size()[-1]):    ##llm:this is only one i
        cfeaturec3 = cfeaturec3[:, :, :, :, i]
        if weight.shape[0] == 104 or weight.shape[0]==100:
            # weight_node = torch.sum(torch.matmul(weight, torch.from_numpy(adj_matrix).cuda()), dim=1)
            cov2 = cov_node(gamma, adj_matrix, cfeaturec3, weight)
            cov_matrix2 = cov2 * cov2
            cov_matrix2 = torch.mean(cov_matrix2, axis=0)  ##sum 数值大于mean
            cov_matrix2 = torch.mean(cov_matrix2, axis=0)
            loss += torch.sum(cov_matrix2) - torch.trace(cov_matrix2)
        elif weight.shape[0] == 12:
            cov3 = cov_T( cfeaturec3, weight)  ## global-local 之后出现负值和NAN(pre is  nan)
            cov_matrix3 = cov3 * cov3
            cov_matrix3 = torch.mean(cov_matrix3, axis=0)  ##sum 数值大于mean
            cov_matrix3 = torch.mean(cov_matrix3, axis=0)
            loss += torch.sum(cov_matrix3) - torch.trace(cov_matrix3)
        else:
            cov1 = cov_sample(cfeaturec3, weight)
            cov_matrix1 = cov1 * cov1    ##* 逐元素相乘, F范数
            # if bool(torch.isnan(cov1).any()):
            #     cov_sample(cfeaturec, weight)
            # with open('results/cov.txt', 'a+') as f:
            #     f.write('cov:'+ str(cov1) + '\n')
            #     f.close()

            # ## squeeze X in advance should exclude the two lines of code
            cov_matrix1 = torch.mean(cov_matrix1, axis = 0)   ##sum 数值大于mean,梯度溢出
            cov_matrix1 = torch.mean(cov_matrix1, axis=0)
            loss += torch.sum(cov_matrix1) - torch.trace(cov_matrix1)
    if weight.shape[0] == 12:
        lambdap = 35
    else:
        lambdap = 35

    lossp = softmax(weight).pow(2).sum()  ##decay_pow=2
    # lambdap = 35 * 1   ## original 70
    # print('loss', loss)
    # print('lossp',lossp)
    loss =  loss / lambdap + lossp

    return loss


def global_local(pre_weight_sample1,pre_features, weight_sample, features, ratio,epoch,i):
    ## llm-global reweighting
    if epoch == 0 and i < 10:
        if features.size()[0] < pre_features.size()[0]:
            pre_features[:features.size()[0]] = (pre_features[:features.size()[0]] * i + features) /(i+1)
            pre_weight_sample1[:features.size()[0]] = (pre_weight_sample1[:features.size()[0]] * i + weight_sample[:features.size()[0]]) / (i+1)
        else:
            pre_features = (pre_features * i + features) / (i + 1)
            pre_weight_sample1 = (pre_weight_sample1 * i + weight_sample) / (i + 1)
    elif features.size()[0] < pre_features.size()[0]:
        pre_features[:features.size()[0]] = pre_features[:features.size()[0]] * ratio + features * (
                    1 - ratio)
        pre_weight_sample1[:features.size()[0]] = pre_weight_sample1[
                                                    :features.size()[0]] * ratio + weight_sample[:features.size()[0]] * (
                                                                1 - ratio)
    else:
        pre_features = pre_features * ratio + features * (1 - ratio)
        pre_weight_sample1 = pre_weight_sample1 * ratio + weight_sample * (1 - ratio)
    return pre_features,pre_weight_sample1
