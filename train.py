import os
import sys
import random
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models import build_model
from utils.validations import validate
from opts import arg_parser
from dataloaders.datasets import build_dataset
from utils.build_cfg import setup_cfg
from dassl.optim import build_lr_scheduler
from utils.trainers import train_coop
from utils.helper import save_checkpoint


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cfg = setup_cfg(args)
   
    # building the train and val dataloaders
    train_split = cfg.DATASET.TRAIN_SPLIT
    val_split = cfg.DATASET.VAL_SPLIT
    test_split = cfg.DATASET.TEST_SPLIT

    train_class = cfg.DATASET.TRAIN_CLASS
    val_class = cfg.DATASET.VAL_CLASS
    test_class = cfg.DATASET.TEST_CLASS
    

    train_loader = build_dataset(cfg, train_split, train_class)
    val_loader = build_dataset(cfg, val_split, val_class)
    test_loader = build_dataset(cfg, test_split, test_class)


    print("input size: ", cfg.INPUT.SIZE)
    # build the model
    model, arch_name = build_model(cfg, args)
    print(arch_name)


    try:
        prompt_params = model.prompt_params()
        mlpcount_params = model.mlpcount_params()
    except:
        prompt_params = model.module.prompt_params()
        mlpcount_params = model.module.mlpcount_params()

    prompt_group = {'params': prompt_params}
    print('num of params in prompt learner: ', len(prompt_params))
    sgd_polices = [prompt_group]
   
    count_group = {'params': mlpcount_params, 'lr': cfg.OPTIM.MLPLR}
    sgd_polices.append(count_group)

    if cfg.TRAINER.FINETUNE_BACKBONE:
        try:
            backbone_params = model.backbone_params()
        except:
            backbone_params = model.module.backbone_params()
        print('num of params in backbone: ', len(backbone_params))
        base_group = {'params': backbone_params, 'lr': cfg.OPTIM.LR * cfg.OPTIM.BACKBONE_LR_MULT}
        sgd_polices.append(base_group)

    if cfg.TRAINER.FINETUNE_ATTN:
        try:
            attn_params = model.attn_params()
        except:
            attn_params = model.module.attn_params()
        print('num of params in attn layer: ', len(attn_params))
        attn_group = {'params': attn_params, 'lr': cfg.OPTIM.LR * cfg.OPTIM.ATTN_LR_MULT}
        sgd_polices.append(attn_group)


    optim = torch.optim.SGD(sgd_polices, lr=cfg.OPTIM.LR,
                                momentum=cfg.OPTIM.MOMENTUM,
                                weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                                dampening=cfg.OPTIM.SGD_DAMPNING,
                                nesterov=cfg.OPTIM.SGD_NESTEROV)
   
    sched = build_lr_scheduler(optim, cfg.OPTIM)
    log_folder = os.path.join(cfg.OUTPUT_DIR, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logfile_path = os.path.join(log_folder, 'log.log')
    if os.path.exists(logfile_path):
        logfile = open(logfile_path, 'a')
    else:
        logfile = open(logfile_path, 'w')

    # logging out some useful information on screen and into log file
    command = " ".join(sys.argv)
    print(command, flush=True)
    print(args, flush=True)
    print(model, flush=True)
    print(cfg, flush=True)
    print(command, file=logfile, flush=True)
    print(args, file=logfile, flush=True)
    print(cfg, file=logfile, flush=True)

  

    if args.auto_resume:
        args.resume = os.path.join(log_folder, 'checkpoint.pth.tar')

    best_mAP = 0
    args.start_epoch = 0
    if args.resume is not None:
        if os.path.exists(args.resume):
            print('... loading pretrained weights from %s' % args.resume)
            print('... loading pretrained weights from %s' % args.resume, file=logfile, flush=True)
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            # TODO: handle distributed version
            best_mAP = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            sched.load_state_dict(checkpoint['scheduler'])

    for epoch in range(args.start_epoch, cfg.OPTIM.MAX_EPOCH):
        batch_time, losses, acc_batches, cacc_batches, \
        miAP_batches, micro_p_batches, micro_r_batches, micro_f_batches, \
        maAP_batches, macro_p_batches, macro_r_batches, macro_f_batches = train_coop(train_loader, model, optim, sched, args, cfg)
        print('Train: [{0}/{1}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {losses.avg:.2f} \t'
              'Acc {acc_batches.avg:.2f} \t'
              'Count Acc {cacc_batches.avg:.2f} \t'
              'miAP {miAP_batches.avg:.2f} \t'
              'Micro P {micro_p_batches.avg:.2f} \t'
              'Micro R {micro_r_batches.avg:.2f} \t'
              'Micro F {micro_f_batches.avg:.2f} \t'
              'maAP {maAP_batches.avg:.2f} \t'
              'Macro P {macro_p_batches.avg:.2f} \t'
              'Macro R {macro_r_batches.avg:.2f} \t'
              'Macro F {macro_f_batches.avg:.2f}'.format(
            epoch + 1, cfg.OPTIM.MAX_EPOCH, batch_time=batch_time,
            losses=losses, acc_batches=acc_batches, cacc_batches=cacc_batches,
            miAP_batches=miAP_batches, micro_p_batches=micro_p_batches, 
            micro_r_batches=micro_r_batches, micro_f_batches=micro_f_batches, 
            maAP_batches=maAP_batches, macro_p_batches=macro_p_batches,
            macro_r_batches=macro_r_batches, macro_f_batches=macro_f_batches), flush=True)

        print('Train: [{0}/{1}]\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {losses.avg:.2f} \t'
              'Acc {acc_batches.avg:.2f} \t'
              'Count Acc {cacc_batches.avg:.2f} \t'
              'miAP {miAP_batches.avg:.2f} \t'
              'Micro P {micro_p_batches.avg:.2f} \t'
              'Micro R {micro_r_batches.avg:.2f} \t'
              'Micro F {micro_f_batches.avg:.2f} \t'
              'maAP {maAP_batches.avg:.2f} \t'
              'Macro P {macro_p_batches.avg:.2f} \t'
              'Macro R {macro_r_batches.avg:.2f} \t'
              'Macro F {macro_f_batches.avg:.2f}'.format(
            epoch + 1, cfg.OPTIM.MAX_EPOCH, batch_time=batch_time,
            losses=losses, acc_batches=acc_batches, cacc_batches=cacc_batches,
            miAP_batches=miAP_batches, micro_p_batches=micro_p_batches, 
            micro_r_batches=micro_r_batches, micro_f_batches=micro_f_batches, 
            maAP_batches=maAP_batches, macro_p_batches=macro_p_batches,
            macro_r_batches=macro_r_batches, macro_f_batches=macro_f_batches), 
            file=logfile, flush=True)

    
        if (epoch + 1) % args.val_every_n_epochs == 0 or epoch == args.stop_epochs - 1:
            acc, c_acc, miAP_score, micro_p, micro_r, micro_f, \
            maAP_score, macro_p, macro_r, macro_f = validate(val_loader, model, args, cfg, split='val')
            
            print('Eval: [{}/{}]\t '
                  'miAP {:.2f} \t P_I {:.2f} \t R_I {:.2f} \t F_I {:.2f} \t maAP {:.2f} \t  P_A {:.2f} \t R_A {:.2f} \t F_A {:.2f} \t Acc {:.2f} \t Count Acc {:.2f}'
                  .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, miAP_score, micro_p, micro_r, micro_f, maAP_score, macro_p, macro_r, macro_f, acc, c_acc), flush=True)
            
            print('Eval: [{}/{}]\t '
                  'miAP {:.2f} \t P_I {:.2f} \t R_I {:.2f} \t F_I {:.2f} \t maAP {:.2f} \t  P_A {:.2f} \t R_A {:.2f} \t F_A {:.2f} \t Acc {:.2f} \t Count Acc {:.2f}'
                  .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, miAP_score, micro_p, micro_r, micro_f, maAP_score, macro_p, macro_r, macro_f, acc, c_acc),
                  file=logfile, flush=True)

            is_best = maAP_score > best_mAP
            if is_best:
                best_mAP = maAP_score
            save_dict = {'epoch': epoch + 1,
                         'arch': arch_name,
                         'state_dict': model.state_dict(),
                         'best_mAP': best_mAP,
                         'optimizer': optim.state_dict(),
                         'scheduler': sched.state_dict()
                         }
            save_checkpoint(save_dict, is_best, log_folder)

    print('Evaluating the best model', flush=True)
    print('Evaluating the best model', file=logfile, flush=True)


    best_checkpoints = os.path.join(log_folder, 'model_best.pth.tar')
    print('... loading pretrained weights from %s' % best_checkpoints, flush=True)
    print('... loading pretrained weights from %s' % best_checkpoints, file=logfile, flush=True)
    checkpoint = torch.load(best_checkpoints, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch']
    
    acc, c_acc, miAP_score, micro_p, micro_r, micro_f, \
    maAP_score, macro_p, macro_r, macro_f = validate(test_loader, model, args, cfg, split='test')
    
    print('Test: [{}/{}]\t '
                  'miAP {:.2f} \t P_I {:.2f} \t R_I {:.2f} \t F_I {:.2f} \t maAP {:.2f} \t  P_A {:.2f} \t R_A {:.2f} \t F_A {:.2f} \t Acc {:.2f} \t Count Acc {:.2f}'
                  .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, miAP_score, micro_p, micro_r, micro_f, maAP_score, macro_p, macro_r, macro_f, acc, c_acc), flush=True)
            
    print('Test: [{}/{}]\t '
            'miAP {:.2f} \t P_I {:.2f} \t R_I {:.2f} \t F_I {:.2f} \t maAP {:.2f} \t  P_A {:.2f} \t R_A {:.2f} \t F_A {:.2f} \t Acc {:.2f} \t Count Acc {:.2f}'
            .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, miAP_score, micro_p, micro_r, micro_f, maAP_score, macro_p, macro_r, macro_f, acc, c_acc),
            file=logfile, flush=True)

if __name__ == '__main__':
    main()