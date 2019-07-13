from __future__ import print_function, division
# import time
import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.nn.functional
import os
import random
import numpy as np
import sklearn.metrics
import traceback
from bladder_dwi_dataset import BladderDwiDataset, MyPreprocessor, MyAugmentation, ToTensor
import collections
import csv
import argparse
import json
import multi_cam_unet
import os.path
import matplotlib.pyplot as plt
import torchvision.models.vgg

def get_iou(x, y):
    x = x.view(x.size()[0], -1)
    y = y.view(y.size()[0], -1)
    i = x & y
    u = x | y
    i = i.float()
    u = u.float()
    sum_i = torch.sum(i, 1)
    sum_u = torch.sum(u, 1)
    iou = (sum_i + 1e-6) / (sum_u + 1e-6)
    return iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_json', type=str, default='cfgs/test.json')
    parser.add_argument('--cv_id', type=str, default='0')
    parser.add_argument('--visual_test', type=str, default='')
    parser.add_argument('--visual_type', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=None)
    return parser.parse_args()


def normalize_images(image):
    image = np.copy(image)
    for i in range(len(image)):
        min = np.min(image[i])
        max = np.max(image[i])
        image[i] = (image[i] - min) / (max - min)
    return image


def transpose(image):
    return image.transpose([0, 3, 2, 1])


def plot_images(images, show_colorbar, name, subtitles=None):
    num_images = len(images)
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / float(rows)))
    vmax = np.max(images)
    vmin = np.min(images)
    f = plt.figure()
    for i in range(num_images):
        ax = f.add_subplot(rows, cols, i + 1)
        ax.axis('off')
        im = ax.imshow(np.squeeze(images[i]), vmin=vmin, vmax=vmax, cmap='gray')
        if subtitles is not None:
            ax.set_title(subtitles[i])
        if show_colorbar:
            f.colorbar(im, ax=ax)
    f.suptitle(name)
    f.show()


def main():
    random.seed()
    using_gpu = torch.cuda.is_available()

    args = parse_args()
    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', args.visual_type)
    print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB', args.batch_size)

    with open(args.cfg_json, 'rb') as fd:
        cfg = json.load(fd)
    cv_id = args.cv_id
    cfg_name = os.path.splitext(os.path.basename(args.cfg_json))[0]
    print(args.cfg_json, cfg_name)
    print(cfg)
    print(cv_id)
    if args.visual_test != '':
        mode = 'visual_test'
        model_weights_path = args.visual_test
    else:
        mode = 'train'
        model_weights_path = ''

    # input dirs
    dataset_csv_dir = cfg['dataset_csv_dir']
    image_root_dir = cfg['image_root_dir']
    cancer_bboxes_root_dir = cfg['cancer_bboxes_root_dir']

    # output dirs
    model_weights_dir = cfg['model_weights_dir']
    log_dir = cfg['log_dir']

    # dataset settings
    num_dataloader_workers = cfg['num_dataloader_workers']
    new_height = cfg['new_height']
    new_width = cfg['new_width']
    using_bladder_mask = cfg['using_bladder_mask']
    caching_data = cfg['caching_data']
    batch_size = cfg['batch_size']

    # model settings
    mil_pooling_type = cfg['mil_pooling_type']
    concat_pred_list = cfg['concat_pred_list']
    num_shared_encoders = cfg['num_shared_encoders']

    # training configurations
    num_step_one_epoches = cfg['num_step_one_epoches']
    num_step_two_epoches = cfg['num_step_two_epoches']
    base_lr = cfg['base_lr']
    loss_weights_list = cfg['loss_weights_list']
    dropout_prob_list = cfg['dropout_prob_list']
    weight_decay = cfg['weight_decay']

    if args.batch_size is not None:
        batch_size = args.batch_size

    model_weights_dir = os.path.join(model_weights_dir, cfg_name)
    if os.path.exists(model_weights_dir):
        assert os.path.isdir(model_weights_dir)
    else:
        os.makedirs(model_weights_dir)

    log_dir = os.path.join(log_dir, cfg_name)
    if os.path.exists(log_dir):
        assert os.path.isdir(log_dir)
    else:
        os.makedirs(log_dir)
    if mode == 'train':
        with open(os.path.join(log_dir, 'cv_{}_cfg.json'.format(cv_id)), 'wb') as fd:
            json.dump(cfg, fd, sort_keys=True, indent=2)

    # prepare dataloaders
    phases = collections.OrderedDict()
    if mode == 'train':
        phases['cv_train'] = os.path.join(dataset_csv_dir, '{}_cv_train.csv'.format(cv_id))
        phases['cv_val'] = os.path.join(dataset_csv_dir, '{}_cv_val.csv'.format(cv_id))
    phases['test'] = os.path.join(dataset_csv_dir, 'test.csv')
    dataloaders = dict()
    for phase in phases:
        csv_path = phases[phase]
        is_training = 'train' in phase
        if 'cv' in phase:
            csv_path = csv_path.format(cv_id)

        preprocessor = MyPreprocessor(image_root_dir, cancer_bboxes_root_dir, new_height, new_width, using_bladder_mask,
                                      True)
        to_tensor = ToTensor()
        if is_training:
            augmentation = MyAugmentation()
            dataset = BladderDwiDataset(csv_path, preprocessor, augmentation, to_tensor, caching_data)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.get_weights(), len(dataset))
        else:
            dataset = BladderDwiDataset(csv_path, preprocessor, None, to_tensor, caching_data)
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler,
                                                 num_workers=num_dataloader_workers, drop_last=is_training)
        dataloaders[phase] = dataloader

    # start training
    model = multi_cam_unet.UNet(1, concat_pred_list, num_shared_encoders, dropout_prob_list)
    if using_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    if model_weights_path != '':
        model.load_state_dict(torch.load(model_weights_path))

    params_to_opt = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params_to_opt, base_lr, weight_decay=weight_decay)

    best_val_iou_cam = 0
    best_val_iou_seg = 0
    best_val_roc_auc = 0
    for epoch in range(num_step_one_epoches + num_step_two_epoches):
        for phase in phases:
            is_training = 'train' in phase
            model.train(is_training)
            loss_cam_list = []
            loss_consistency_list = []
            iou_cam_list = []
            iou_seg_list = []
            score_array_list = []
            label_array_list = []

            for step, data in enumerate(dataloaders[phase]):
                image = data['image']
                label = data['label']
                cancer_bboxes_image = data['cancer_bboxes_image']
                if using_gpu:
                    image = image.cuda()
                    label = label.cuda()
                    cancer_bboxes_image = cancer_bboxes_image.cuda()
                image = torch.autograd.Variable(image, volatile=not is_training)
                label = torch.autograd.Variable(label)
                preds_tuple = model(image)

                losses = []
                # Loss_CAM
                for cam in preds_tuple[:-1]:
                    if mil_pooling_type == 'max':
                        score = torch.nn.functional.adaptive_max_pool2d(cam, (1, 1)).view(-1, 1)
                    elif mil_pooling_type == 'avg':
                        score = torch.nn.functional.adaptive_avg_pool2d(cam, (1, 1)).view(-1, 1)
                    else:
                        raise Exception('Unknown mil_pooling_type')
                    loss_cam = torch.nn.functional.binary_cross_entropy_with_logits(score, label)
                    loss_cam_value = loss_cam.data[0]
                    losses.append(loss_cam)

                # upsample the last cam to get the pseudo label
                pseudo_label = torch.nn.functional.upsample(cam, [new_height, new_width], mode='bilinear') > 0
                pseudo_label = pseudo_label.float()

                # Loss_Consistency
                score_map = preds_tuple[-1]
                loss_consistency = torch.nn.functional.binary_cross_entropy_with_logits(score_map, pseudo_label)
                loss_consistency_value = loss_consistency.data[0]
                if epoch >= num_step_one_epoches:
                    losses.append(loss_consistency)

                # get total loss
                total_loss = 0.0
                for i, loss in enumerate(losses):
                    if loss_weights_list[i] != 0.0:
                        total_loss += loss_weights_list[i] * loss

                # optimize
                if is_training:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                # summary of this step
                iou_cam = get_iou(pseudo_label.data > 0, cancer_bboxes_image > 0)
                iou_cam_mean = torch.sum(iou_cam * label.data.squeeze()) / (torch.sum(label.data.squeeze()) + 1e-6)
                iou_seg = get_iou(score_map.data > 0, cancer_bboxes_image > 0)
                iou_seg_mean = torch.sum(iou_seg * label.data.squeeze()) / (
                    torch.sum(label.data.squeeze()) + 1e-6)
                confusion_matrix = sklearn.metrics.confusion_matrix(torch.gt(label.data.cpu(), 0.5),
                                                                    torch.gt(score.data.cpu(), 0))

                # report of this step
                # print(
                #     'Epoch {:>3}, Phase {}, Step {:>4}, Loss_CAM={:.4f}, Loss_Consistency={:.4f}, IOU_CAM={:.4f}, IOU_SEG={:.4f}'.format(
                #         epoch, phase, step,
                #         loss_cam_value,
                #         loss_consistency_value, iou_cam_mean, iou_seg_mean))
                # print(confusion_matrix)
                # for loss in losses:
                #     print(loss.data[0], end=' ')
                # print()

                # summary of this epoch
                score_array_list.append(score.data.cpu().squeeze().numpy())
                label_array_list.append(label.data.cpu().squeeze().numpy())
                loss_cam_list.append(loss_cam_value)
                loss_consistency_list.append(loss_consistency_value)
                iou_cam_list.append(iou_cam.cpu().numpy())
                iou_seg_list.append(iou_seg.cpu().numpy())
                if mode == 'visual_test':
                    accession_number = np.array(data['accession_number'])
                    correct = (score.data.cpu().numpy() > 0) == label.data.cpu().numpy()
                    correct = np.squeeze(correct)
                    wrong = np.logical_not(correct)
                    # plot_idx = np.ones_like(correct, dtype=np.bool)
                    plot_idx = wrong
                    print(args.visual_type)
                    if args.visual_type=='wrong':
                        plot_idx = wrong
                    elif args.visual_type=='correct':
                        plot_idx = correct
                    else:
                        plot_idx = np.ones_like(correct, dtype=np.bool) #all
                    image = transpose(image.data.cpu().numpy())
                    # plot_images(normalize_images(image[plot_idx, :, :, 0]), False, "ADC", accession_number[plot_idx])
                    plot_images(normalize_images(image[plot_idx, :, :, 1]), False, "B=0", accession_number[plot_idx])
                    plot_images(normalize_images(image[plot_idx, :, :, 2]), False, "B=1000", accession_number[plot_idx])
                    plot_images(transpose(cancer_bboxes_image.cpu().numpy())[plot_idx], False, "GT", accession_number[plot_idx])
                    # plot_images(transpose(cam.data.cpu().numpy())[plot_idx], False, "CAM", accession_number[plot_idx])
                    plot_images(transpose(pseudo_label.data.cpu().numpy())[plot_idx], False, "Prediction_CAM", accession_number[plot_idx])
                    # plot_images(transpose(score_map.data.cpu().numpy())[plot_idx], False, "score_map", accession_number[plot_idx])
                    # plot_images(transpose((score_map > 0).data.cpu().numpy())[plot_idx], False, "score_map>0", accession_number[plot_idx])
                    plt.show()

            # report of this epoch
            loss_cam = np.mean(loss_cam_list)
            loss_consistency = np.mean(loss_consistency_list)
            score = np.concatenate(score_array_list)
            label = np.concatenate(label_array_list).astype(np.int)
            iou_cam = np.concatenate(iou_cam_list)
            iou_cam = np.sum(iou_cam * label) / (np.sum(label) + 1e-6)
            iou_seg = np.concatenate(iou_seg_list)
            iou_seg = np.sum(iou_seg * label) / (np.sum(label) + 1e-6)
            confusion_matrix = sklearn.metrics.confusion_matrix(label, np.greater(score, 0))
            roc_auc = sklearn.metrics.roc_auc_score(label, score)
            print(
                'Epoch {:>3}, Phase {} Complete! Loss_CAM={:.4f}, Loss_Consistency={:.4f}, IOU_CAM={:.4f}, IOU_SEG={:.4f}, ROC_AUC={:.4f}'
                    .format(epoch, phase, loss_cam, loss_consistency, iou_cam, iou_seg, roc_auc))
            print(confusion_matrix)

            if mode == 'train':
                try:
                    # saving log
                    with open(os.path.join(log_dir, "{}_{}.csv".format(cv_id, phase)), 'ab') as fd:
                        csv.writer(fd).writerow([epoch, phase, loss_cam, loss_consistency, iou_cam, iou_seg, roc_auc])
                except:
                    traceback.print_exc()
                if is_training:
                    try:
                        torch.save(model.state_dict(), os.path.join(model_weights_dir, "cv_{}_last.pth".format(cv_id)))
                    except:
                        traceback.print_exc()
                if 'val' in phase:
                    if roc_auc > best_val_roc_auc:
                        best_val_roc_auc = roc_auc
                        print('New best_val_roc_auc: {:.4f}, Epoch {}'.format(best_val_roc_auc, epoch))
                        try:
                            with open(os.path.join(log_dir, "best_val_roc_auc.txt"), 'w') as fd:
                                fd.write(str(best_val_roc_auc))
                            torch.save(model.state_dict(),
                                       os.path.join(model_weights_dir, "cv_{}_best_roc_auc.pth".format(cv_id)))
                        except:
                            traceback.print_exc()
                    if iou_cam > best_val_iou_cam:
                        best_val_iou_cam = iou_cam
                        print('New best_val_iou_cam: {:.4f}, Epoch {}'.format(best_val_iou_cam, epoch))
                        try:
                            with open(os.path.join(log_dir, "best_val_iou_cam.txt"), 'w') as fd:
                                fd.write(str(best_val_iou_cam))
                            torch.save(model.state_dict(),
                                       os.path.join(model_weights_dir, "cv_{}_best_iou_cam.pth".format(cv_id)))
                        except:
                            traceback.print_exc()
                    if iou_seg > best_val_iou_seg:
                        best_val_iou_seg = iou_seg
                        print('New best_val_iou_seg: {:.4f}, Epoch {}'.format(best_val_iou_seg, epoch))
                        try:
                            with open(os.path.join(log_dir, "best_val_iou_seg.txt"), 'w') as fd:
                                fd.write(str(best_val_iou_seg))
                            torch.save(model.state_dict(),
                                       os.path.join(model_weights_dir, "cv_{}_best_iou_seg.pth".format(cv_id)))
                        except:
                            traceback.print_exc()


if __name__ == '__main__':
    main()
