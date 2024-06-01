
from torchvision import transforms
import torchvision
import os
from .data_loader_multi import EpisodicDataset, KShotTaskSampler

ITER_TABLE = {
    'train': 100,
    'val': 100,
    'test': 200,
}




def get_dataset(dataset_name, 
                data_root,
                split, 
                classes,
                support_size,
                query_size, 
                n_iters):

    if dataset_name == "episodic_coco":
        file_path = os.path.join(data_root, split+'/labels.json')
        dataset = EpisodicDataset(file_path)
        sampler = KShotTaskSampler(dataset, 
                                    episodes_per_epoch=n_iters, 
                                    n=support_size, 
                                    k=classes, 
                                    q=query_size, 
                                    num_tasks=1)

    elif dataset_name == "episodic_voc":
        file_path = os.path.join(data_root, split+'/labels.json')
        dataset = EpisodicDataset(file_path)
        sampler = KShotTaskSampler(dataset, 
                                    episodes_per_epoch=n_iters, 
                                    n=support_size, 
                                    k=classes, 
                                    q=query_size, 
                                    num_tasks=1)
    return sampler


def build_dataset(cfg, data_split, class_count):
    print(' -------------------- Building Dataset ----------------------')
    print('DATASET.ROOT = %s' % cfg.DATASET.ROOT)
    print('data_split = %s' % data_split)
   
    try:
        if 'train' in data_split or 'Train' in data_split:
            img_size = cfg.INPUT.TRAIN.SIZE[0]
        else:
            img_size = cfg.INPUT.TEST.SIZE[0]
    except:
        img_size = cfg.INPUT.SIZE[0]
    print('INPUT.SIZE = %d' % img_size)

    
    return get_dataset(dataset_name=cfg.DATASET.NAME, 
                        data_root=cfg.DATASET.ROOT,
                        split=data_split, 
                        classes=class_count,
                        support_size=1,
                        query_size=4, 
                        n_iters=ITER_TABLE[data_split])

