from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
import medicalDataLoader
from utils import *
from IVD_Net import *
import time
from optimizer import Adam
import os
import csv

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

    
def runTraining():
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = 2
    batch_size_val = 1
    batch_size_val_save = 1
 
    lr = 0.0001
    epoch = 200
    num_classes = 2
    initial_kernels = 32
    
    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/0_label_1_train_val.csv'
    val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/label_1_test.csv'
    data_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'
    label_path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series_labels'
    train = []
    val = []
    
    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:

            train.append(i)

    with open(val_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:

            val.append(i)

    modelName = 'IVD_Net'
    
    
    img_names_ALL = []
    print('.'*40)
    print(" ....Model name: {} ........".format(modelName))
    
    print(' - Num. classes: {}'.format(num_classes))
    print(' - Num. initial kernels: {}'.format(initial_kernels))
    print(' - Batch size: {}'.format(batch_size))
    print(' - Learning rate: {}'.format(lr))
    print(' - Num. epochs: {}'.format(epoch))

    print('.'*40)
    root_dir = '../Data/Training_PngITK'
    model_dir = 'IVD_Net'


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = medicalDataLoader.MedicalImageDataset('train',
                                                      train,
                                                      data_path,
                                                      label_path,
                                                      160, 160,
                                                      transform=transform,
                                                      mask_transform=mask_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=5,
                              shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    val,
                                                    data_path,
                                                    label_path,
                                                    160, 160,
                                                    transform=transform,
                                                    mask_transform=mask_transform)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=5,
                            shuffle=False)
    
    val_loader_save_images = DataLoader(val_set,
                                        batch_size=batch_size_val_save,
                                        num_workers=5,
                                        shuffle=False)
    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")

    net = IVD_Net_asym(1,num_classes,initial_kernels)
    
    # Initialize the weights
    net.apply(weights_init)

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_ = computeDiceOneHotBinary()

    if torch.cuda.is_available():
        net.cuda()
        softMax.cuda()
        CE_loss.cuda()
        Dice_.cuda()
        
    # To load a pre-trained model
    '''try:
        net = torch.load('modelName')
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''
        
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           patience=4,
                                                           verbose=True,
                                                           factor=10 ** -0.5)
 
    BestDice, BestEpoch = 0, 0

    d1Train = []
    d1Val = []
    Losses = []

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        net.train()
        lossTrain = []
        d1TrainTemp = []
        
        totalImages = len(train_loader)
        
        for j, data in enumerate(train_loader):
            
            image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs, labels = data
            
            # Be sure your data here is between [0,1]
            image_ADC=image_ADC.type(torch.FloatTensor)
            image_b0=image_b0.type(torch.FloatTensor)
            image_b1000=image_b1000.type(torch.FloatTensor)
            image_t2w=image_t2w.type(torch.FloatTensor)
            image_t2wfs=image_t2wfs.type(torch.FloatTensor)
            
            labels = labels.numpy()
            idx=np.where(labels>0.0)
            labels[idx]=1.0
            labels = torch.from_numpy(labels)
            labels = labels.type(torch.FloatTensor)
          
            optimizer.zero_grad()
            MRI = to_var(torch.cat((image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs),dim=1))
            
            Segmentation = to_var(labels)
            
            target_dice = to_var(torch.ones(1))

            net.zero_grad()
            
            segmentation_prediction = net(MRI)
            predClass_y = softMax(segmentation_prediction)
            
            Segmentation_planes = getOneHotSegmentation(Segmentation)
            segmentation_prediction_ones = predToSegmentation(predClass_y)

            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)

            CE_loss_ = CE_loss(segmentation_prediction, Segmentation_class)

            # Compute the Dice (so far in a 2D-basis)
            DicesB, DicesF = Dice_(segmentation_prediction_ones, Segmentation_planes)
            DiceB = DicesToDice(DicesB)
            DiceF = DicesToDice(DicesF)
            
            loss = CE_loss_ 
            
            loss.backward()
            optimizer.step()
            
            lossTrain.append(loss.item())
 
            #printProgressBar(j + 1, totalImages, prefix="[Training] Epoch: {} ".format(i), length=15, suffix=" Mean Dice: {:.4f},".format(DiceF.item()))

        printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f}".format(i,np.mean(lossTrain)))
        # Save statistics
        Losses.append(np.mean(lossTrain))
        d1 = inference(net, val_loader, batch_size, i)
        d1Val.append(d1)
        d1Train.append(np.mean(d1TrainTemp).item())
        
        mainPath = '../Results/Statistics/' + modelName
        
        directory = mainPath
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, 'Losses.npy'), Losses)
        np.save(os.path.join(directory, 'd1Val.npy'), d1Val)
        np.save(os.path.join(directory, 'd1Train.npy'), d1Train)
       
        currentDice = d1[0].numpy()

        print("[val] DSC: {:.4f} ".format(d1[0]))
        
          
        if currentDice > BestDice:
            BestDice = currentDice
     
            BestEpoch = i
            if currentDice > 0.75:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(net, os.path.join(model_dir, "Best_" + modelName + ".pkl"))
                saveImages(net, val_loader_save_images, batch_size_val_save, i, modelName)
        
        print("best DSC", BestDice)
        # Two ways of decay the learning rate:      
        if i % (BestEpoch + 10):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        #scheduler.step(currentDice)

if __name__ == '__main__':
    runTraining()
