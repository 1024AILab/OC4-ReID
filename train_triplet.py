# coding=utf-8
# @FileName:train_triplet.py
# @Time:2024/7/27 
# @Author: CZH
# coding=utf-8
# @FileName:train2.py
# @Time:2024/7/24
# @Author: CZH
import time
import datetime
import logging
import torch
from apex import amp
from tools.utils import AverageMeter


def train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair,
              criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_cla_loss1 = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_pair_loss1 = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features, f, f1, f2, f3, f4, f5, f6 = model(imgs)
        # features.shape, f2.shape torch.Size([64, 4096]) torch.Size([64, 150])
        # print("features.shape, f2.shape", features.shape, f2.shape)
        outputs = classifier(features)
        # print(outputs.shape)
        pred_clothes = clothes_classifier(features.detach())
        _, preds = torch.max(outputs.data, 1)
        _, preds1 = torch.max(f.data, 1)

        # Update the clothes discriminator
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            if config.TRAIN.AMP:
                with amp.scale_loss(clothes_loss, optimizer_cc) as scaled_loss:
                    scaled_loss.backward()
            else:
                clothes_loss.backward()
            optimizer_cc.step()

        # Update the backbone
        new_pred_clothes = clothes_classifier(features)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        # print("cla_loss", cla_loss, type(cla_loss.item()))
        cla_loss1 = criterion_cla(f, pids)
        # print("cla_loss1", cla_loss1, type(cla_loss1.item()))
        pair_loss = criterion_pair(features, pids)
        pair_loss_tmp = criterion_pair(f1, pids) + criterion_pair(f2, pids) + criterion_pair(f3, pids) + \
                        criterion_pair(f4, pids) + criterion_pair(f5, pids) + criterion_pair(f6, pids)
        pair_loss_tmp = pair_loss_tmp / 6
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + 0.01 * pair_loss_tmp + 0.01 * cla_loss1 + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
        else:
            loss = cla_loss + 0.01 * pair_loss_tmp + 0.01 * cla_loss1 + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float() / clothes_ids.size(0),
                                clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_cla_loss1.update(cla_loss1.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_pair_loss1.update(pair_loss_tmp.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'ClaLoss id :{cla_loss1.avg:.4f} '
                'PairLoss Triplet:{pair_loss1.avg:.4f} '
                'CloLoss:{clo_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '
                'CloAcc:{clo_acc.avg:.2%} '.format(
        epoch + 1, batch_time=batch_time, data_time=data_time,
        cla_loss=batch_cla_loss, cla_loss1=batch_cla_loss1,
        pair_loss=batch_pair_loss, pair_loss1=batch_pair_loss1,
        clo_loss=batch_clo_loss, adv_loss=batch_adv_loss,
        acc=corrects, clo_acc=clothes_corrects))


def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair,
                          criterion_adv, optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        if epoch >= config.TRAIN.START_EPOCH_ADV:
            adv_loss = criterion_adv(features, clothes_ids, pos_mask)
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss

        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
        epoch + 1, batch_time=batch_time, data_time=data_time,
        cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
        adv_loss=batch_adv_loss, acc=corrects))
# coding=utf-8
# @FileName:train1.py
# @Time:2024/7/24
# @Author: CZH
