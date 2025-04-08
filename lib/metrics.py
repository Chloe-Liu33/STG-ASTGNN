# -*- coding:utf-8 -*-

import numpy as np
import torch


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.greater_equal(y_true, 1e-3)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mape(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):  ##return true if an element is NaN
            mask = ~torch.isnan(labels)  ## return a new tensor with boolean elements representing
            # if each elemetns of the input  is NaN or not.
            # print("this is isnan")   #不走这条路
        else:
            mask = (labels != null_val)   ## shape is  120,100,12,1   tensor
            # print('this is else',mask.shape)
        mask = mask.float()
        mask /= torch.mean(mask)
        # mask1 = mask.cpu().numpy()
        # mask1 = tuple(map(tuple,mask1))## tensor can not be used to torch.zeros
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask),mask)  ##return a tensor of elements selected from x and y,depending on condition
        loss = torch.abs((preds - labels) / labels)
        loss = torch.nan_to_num(loss*mask)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return loss
def masked_mape2(y_true, y_pred, null_val=np.nan):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      np.clip(y_true,100,600)))       ## To what extent clip the denominator is important???
        mape = np.nan_to_num(mask * mape)
        mape = torch.from_numpy(mape).cuda()
        return mape