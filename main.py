"""
 Between-class Learning for Image Classification.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
import chainer

import csv
# import matplotlib.pyplot as plt
import numpy as np

import opts
import models
import dataset
from train import Trainer


def main():
    opt = opts.parse()
    chainer.cuda.get_device_from_id(opt.gpu).use()
    for i in range(1, opt.nTrials + 1):
        print('+-- Trial {} --+'.format(i))
        t_err, v_err = train(opt, i)

    return t_err, v_err


def train(opt, trial):
    model = getattr(models, opt.netType)(opt.nClasses)
    model.to_gpu()
    optimizer = chainer.optimizers.NesterovAG(lr=opt.LR, momentum=opt.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(opt.weightDecay))
    train_iter, val_iter = dataset.setup(opt)
    trainer = Trainer(model, optimizer, train_iter, val_iter, opt)

    t_err = []
    v_err = []

    for epoch in range(1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f}\n'.format(
                epoch, opt.nEpochs, trainer.optimizer.lr, train_loss, train_top1, val_top1))
        sys.stdout.flush()
        
        t_err = np.append(t_err, train_top1)
        v_err = np.append(v_err, val_top1)
        
        
    if opt.save != 'None':
        chainer.serializers.save_npz(
            os.path.join(opt.save, 'model_trial{}.npz'.format(trial)), model)
            
    return t_err, v_err


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3) #设精度为3
    # np.savetxt('data/submit.txt', res, fmt='%.03f') #保留3位小数
    err = main()
    print(err)
    currentpath = '/content/drive/My Drive/DL'
    # currentpath = os.getcwd()
    
    with open(currentpath + "/file.txt", 'wb') as f:
        np.savetxt(f, err, fmt = '%.03f')
    
   # f = open(currentpath + "/file.txt","a")
#    f.write(err)

    
