import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
from progressBar import printProgressBar
import scipy.io as sio
from scipy import ndimage


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class computeDiceOneHotBinary(nn.Module):
    def __init__(self):
        super(computeDiceOneHotBinary, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceF = to_var(torch.zeros(batchsize, 2))
        
        for i in range(batchsize):
            DiceB[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceF[i, 0] = self.inter(pred[i, 1], GT[i, 1])
           
            DiceB[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceF[i, 1] = self.sum(pred[i, 1], GT[i, 1])
           
        return DiceB, DiceF 
        
        
def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def getSingleImageBin(pred):
    # input is a 2-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    n_channels = 2
    Val = to_var(torch.zeros(2))
    Val[1] = 1.0
    
    x = predToSegmentation(pred)
    out = x * Val.view(1, n_channels, 1, 1)
    return out.sum(dim=1, keepdim=True)
    

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()



def getOneHotSegmentation(batch):
    backgroundVal = 0
    # IVD
    label1 = 1.0
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1), 
                             dim=1)
                             
    return oneHotLabels.float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    spineLabel = 1.0
    return (batch / spineLabel).round().long().squeeze()


def saveImages(net, img_batch, batch_size, epoch, modelName):
    path = '../Results/Images_PNG/' + modelName + '_'+ str(epoch) 
    if not os.path.exists(path):
        os.makedirs(path)
        
    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()
    
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image_f,image_i,image_o,image_w, labels, img_names = data

        # Be sure here your image is betwen [0,1]
        image_f=image_f.type(torch.FloatTensor)
        image_i=image_i.type(torch.FloatTensor)
        image_o=image_o.type(torch.FloatTensor)
        image_w=image_w.type(torch.FloatTensor)

        images = torch.cat((image_f,image_i,image_o,image_w),dim=1)

        MRI = to_var(images)
        image_f_var = to_var(image_f)
        Segmentation = to_var(labels)
            
        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImageBin(pred_y)
        imgname = img_names[0].split('/Fat/')
        imgname = imgname[1].split('_fat.png')
        
        out = torch.cat((image_f_var, segmentation, Segmentation*255))
        
        torchvision.utils.save_image(out.data, os.path.join(path,imgname[0] + '.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False)
                                     
    printProgressBar(total, total, done="Images saved !")
   
    
def inference(net, img_batch, batch_size, epoch):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    net.eval()
    
    dice = computeDiceOneHotBinary().cuda()
    softMax = nn.Softmax().cuda()

    #img_names_ALL = []
    for i, data in enumerate(img_batch):
        #printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        #image_f,image_i,image_o,image_w, labels, img_names = data
        image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs, labels = data

        # Be sure here your image is betwen [0,1]
        '''
        image_f=image_f.type(torch.FloatTensor)/65535
        image_i=image_i.type(torch.FloatTensor)/65535
        image_o=image_o.type(torch.FloatTensor)/65535
        image_w=image_w.type(torch.FloatTensor)/65535
        '''
        image_ADC=image_ADC.type(torch.FloatTensor)
        image_b0=image_b0.type(torch.FloatTensor)
        image_b1000=image_b1000.type(torch.FloatTensor)
        image_t2w=image_t2w.type(torch.FloatTensor)
        image_t2wfs=image_t2wfs.type(torch.FloatTensor)

        images = torch.cat((image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs),dim=1)
        #images = torch.cat((image_f,image_i,image_o,image_w),dim=1)
        #img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(images)

        labels = labels.numpy()
        idx=np.where(labels>0.0)
        labels[idx]=1.0
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.FloatTensor)
  
        Segmentation = to_var(labels)
        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        
        Segmentation_planes = getOneHotSegmentation(Segmentation)
        segmentation_prediction_ones = predToSegmentation(pred_y)
        
        DicesN, Dices1 = dice(segmentation_prediction_ones, Segmentation_planes)

        Dice1[i] = Dices1.data
        

    printProgressBar(total, total, done="[Inference] Segmentation Done !")
    
    ValDice1 = DicesToDice(Dice1)
   
    return [ValDice1]






