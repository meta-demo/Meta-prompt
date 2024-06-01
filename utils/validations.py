import sys
sys.path.insert(0, '../')
import torch
import time
from utils.helper import AverageMeter, mAP, calc_F1
from torch.cuda.amp import autocast
from utils.asymmetric_loss import ProtypicalLoss
from utils.trainers import obtain_img_labels

def validate(data_loader, model, args, cfg, split):

    batch_time = AverageMeter()
   
    miAP_batches = AverageMeter()
    maAP_batches = AverageMeter()

    losses = AverageMeter()
    acc_batches = AverageMeter()
    cacc_batches = AverageMeter()
    macro_p_batches, macro_r_batches, macro_f_batches = AverageMeter(), AverageMeter(), AverageMeter()
    micro_p_batches, micro_r_batches, micro_f_batches = AverageMeter(), AverageMeter(), AverageMeter()

    model.eval()

    criterion = ProtypicalLoss(cfg.TRAINER.DEVICEID, cfg.TRAINER.BETA, cfg.TRAINER.GAMMA)
    
    root = cfg.DATASET.ROOT 
    

    
    with torch.no_grad():
        end = time.time()

        for i, (batch) in enumerate(data_loader):
    
            support, query, classnames = batch
           
            support_img, query_img, support_ids, query_ids = obtain_img_labels(root, split, support, query, classnames, args.input_size)
            
            support_size = support_img.shape[0]
            images = torch.cat((support_img, query_img), dim=0)
            target = torch.cat((support_ids, query_ids), dim=0)
           
            if torch.cuda.is_available():
                device = torch.device("cuda", cfg.TRAINER.DEVICEID)
            else:
                device = torch.device("cpu")
            images = images.to(device)
            target = target.to(device)
            
            # compute output
            with autocast():
                image_features, text_features, count_outputs, weights = model(classnames, images)
            
           
            loss, acc, c_acc, mi_ap, micro_p, micro_r, micro_f, \
            ma_ap, macro_p, macro_r, macro_f = criterion(image_features, text_features, count_outputs, weights, support_size, target)

            losses.update(loss.item(), 1)

            print(f"Loss: {loss}, Acc: {acc}, Count acc: {c_acc}")
            print(f"Macro mAP:{ma_ap}, Macro P: {macro_p}, Macro R: {macro_r}, Macro F1: {macro_f}")
            print(f"Micro mAP:{mi_ap}, Micro P: {micro_p}, Micro R: {micro_r}, Micro F1: {micro_f}")

            miAP_batches.update(mi_ap)
            maAP_batches.update(ma_ap)
            acc_batches.update(acc)
            cacc_batches.update(c_acc)
            micro_p_batches.update(micro_p)
            micro_r_batches.update(micro_r)
            micro_f_batches.update(micro_f)
            macro_p_batches.update(macro_p)
            macro_r_batches.update(macro_r)
            macro_f_batches.update(macro_f)
            
            batch_time.update(time.time()-end)
            end = time.time()
       
   
    miAP_score = miAP_batches.avg
    maAP_score = maAP_batches.avg
    acc = acc_batches.avg
    c_acc = cacc_batches.avg
    macro_p = macro_p_batches.avg
    macro_r = macro_r_batches.avg
    macro_f = macro_f_batches.avg
    micro_p = micro_p_batches.avg
    micro_r = micro_r_batches.avg
    micro_f = micro_f_batches.avg

    
    return acc, c_acc, miAP_score, micro_p, micro_r, micro_f, \
            maAP_score, macro_p, macro_r, macro_f
