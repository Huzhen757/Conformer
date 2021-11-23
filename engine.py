"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import ModelEma

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True
                    ):
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)  # 原始的一个batch size的数据
        targets = targets.to(device, non_blocking=True)  # 原始的一个batch size数据对应的target
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, device=device) # 经过Mixup之后得到的一组新的数据以及对应的label

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if isinstance(outputs, list): # 卷积分支的预测值与GT进行softTargetCEloss再/2,transformer分支同理
                loss_list = [criterion(o, targets) / len(outputs) for o in outputs] 
                loss = sum(loss_list) # 两个分支的loss进行累加
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        
        if isinstance(outputs, list):
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
        else:
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # Conformer
            if isinstance(output, list):
                loss_list = [criterion(o, target) / len(output)  for o in output]
                loss = sum(loss_list)
            # others
            else:
                loss = criterion(output, target)
        if isinstance(output, list):
            # Conformer
            acc1_head1 = accuracy(output[0], target, topk=(1,))[0]  # 卷积分支的top1 acc
            acc1_head2 = accuracy(output[1], target, topk=(1,))[0]  # transformer分支的top1 acc
            acc1_total = accuracy(output[0] + output[1], target, topk=(1,))[0] # 两者预测值相加之后的acc
        else:
            # others
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        if isinstance(output, list):
            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
            metric_logger.meters['acc1'].update(acc1_total.item(), n=batch_size)
            metric_logger.meters['acc1_head1'].update(acc1_head1.item(), n=batch_size)
            metric_logger.meters['acc1_head2'].update(acc1_head2.item(), n=batch_size)
        else:
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if isinstance(output, list):
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} Acc@head_1 {head1_top1.global_avg:.3f} Acc@head_2 {head2_top1.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} loss@1 {loss_0.global_avg:.3f} loss@2 {loss_1.global_avg:.3f} '
              .format(heads_top1=metric_logger.acc1, head1_top1=metric_logger.acc1_head1, head2_top1=metric_logger.acc1_head2,
                      losses=metric_logger.loss, loss_0=metric_logger.loss_0, loss_1=metric_logger.loss_1))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk) # topk((1,)) 表示top1准确率; 若topk((1,5)) 则表示取top1和top5准确率
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True) # 取output中的最大值，按照维度=1的位置取也就是按行取最大值
    # topk中输出䣌第一个值为该tensor中的最大值，第二个值返回的是该最大值所在的索引
    pred = pred.t() # 最大值索引进行转置，若batchsize=8，输出的pred.shape=(8,1) -> (1, 8)
    correct = pred.eq(target.reshape(1, -1).expand_as(pred)) # target扩展一个维度，其shape与pred一致 (1, 8)
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    # 使用eq，匹配pred与target相等的值，若相等则为True否则为False e.g. [True,False,True,False,True,True,True,False]
    # 遍历topk中的值，topk=1那么就是计算top-1 acc，则取第一个值，除上其batchsize，获取准确率; 
    # 若topk=5，则获取前5个值，reshape:(1, 5) -> (5) , 5个值进行加和除上其batchsize，获取top-5 acc
