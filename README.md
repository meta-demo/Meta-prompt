# Meta-Prompt Tuning Vision-Language Model for Multi-Label Few-Shot Image Recognition

Thanks for your attention. The following instructions can help you reproduce the experiments.

## Platform

Our experiments are conducted on a platform with NVIDIA GeForce RTX 3090.


## Set-up Experiment Environment

Our implementation is in Pytorch with python 3.7. 

And follow [the link](https://github.com/guozix/TaI-DPT) to install `dassl` and `clip`.

## Datasets

- **MS COCO**: We include images from the official `train2014` and `val2014` splits.
- **PASCAL VOC**: We include images from the official `trainval` and `test` splits of VOC2007 and `trainval` of VOC2012. 



## Running

```
bash run.sh
```

The detailed configurations can be found in the ```run.sh``` and ```opts.py```.


Some Args:  
- `dataset_config_file`: currently the code supports `configs/datasets/coco.yaml` and `configs/datasets/voc.yaml`.
- `lr`: learning rate.
- `n_ctx`: length of each prompt.
- `pool_size`: number of learnable prompts.
  
