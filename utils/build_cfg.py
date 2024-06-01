from dassl.config import get_cfg_default


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.MLCCLIP = CN()
    
    cfg.MLCCLIP.FLOAT = False
    cfg.TRAINER.COOP_MLC = CN()
    cfg.TRAINER.COOP_MLC.N_CTX = 16
    cfg.TRAINER.COOP_MLC.CSC = False
    cfg.TRAINER.COOP_MLC.POOL_SIZE = 8
    cfg.TRAINER.COOP_MLC.PROMPT_KEY_INIT = ""
    

    cfg.TRAINER.RESNET_IMAGENET = CN()
    cfg.TRAINER.RESNET_IMAGENET.DEPTH = 50
    cfg.TRAINER.DEVICEID = 0
    cfg.TRAINER.BETA = 0.1
    cfg.TRAINER.GAMMA = 0.1
    cfg.TRAINER.FINETUNE = False
    cfg.TRAINER.FINETUNE_BACKBONE = False
    cfg.TRAINER.FINETUNE_ATTN = False

    cfg.DATASET.VAL_SPLIT = ""
    cfg.DATASET.VAL_CLASS = 1
    cfg.DATASET.TEST_SPLIT = ""
    cfg.DATASET.TEST_CLASS = 1
    cfg.DATASET.TRAIN_SPLIT = ""
    cfg.DATASET.TRAIN_CLASS = 1
    
  
    cfg.INPUT.TRAIN = CN()
    cfg.INPUT.TRAIN.SIZE = (224, 224)
    cfg.INPUT.TEST = CN()
    cfg.INPUT.TEST.SIZE = (224, 224)
    cfg.OPTIM.MLPLR = 0.0001
    cfg.OPTIM.BACKBONE_LR_MULT = cfg.OPTIM.BASE_LR_MULT
    cfg.OPTIM.ATTN_LR_MULT = cfg.OPTIM.BASE_LR_MULT


def reset_cfg(cfg, args):
    
   

    if args.datadir:
        cfg.DATASET.ROOT = args.datadir

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.print_freq:
        cfg.TRAIN.PRINT_FREQ = args.print_freq

    if args.input_size:
        cfg.INPUT.SIZE = (args.input_size, args.input_size)
        cfg.INPUT.TRAIN.SIZE = (args.input_size, args.input_size)
        cfg.INPUT.TEST.SIZE = (args.input_size, args.input_size)

    if args.train_input_size:
        cfg.INPUT.TRAIN.SIZE = (args.train_input_size, args.train_input_size)
        cfg.INPUT.SIZE = (args.train_input_size, args.train_input_size)

    if args.test_input_size:
        cfg.INPUT.TEST.SIZE = (args.test_input_size, args.test_input_size)

    if args.lr:
        cfg.OPTIM.LR = args.lr

    if args.mlplr:
        cfg.OPTIM.MLPLR = args.mlplr

    if args.device_id:
        cfg.TRAINER.DEVICEID = args.device_id

    if args.beta:
        cfg.TRAINER.BETA = args.beta

    if args.gamma:
        cfg.TRAINER.GAMMA = args.gamma
        
    if args.csc:
        cfg.TRAINER.COOP_MLC.CSC = args.csc

    if args.n_ctx:
        cfg.TRAINER.COOP_MLC.N_CTX = args.n_ctx

    if args.logit_scale:
        cfg.TRAINER.COOP_MLC.LS = args.logit_scale

    if args.pool_size:
        cfg.TRAINER.COOP_MLC.POOL_SIZE = args.pool_size

    if args.prompt_key_init:
        cfg.TRAINER.COOP_MLC.PROMPT_KEY_INIT = args.prompt_key_init
        
    if args.finetune:
        cfg.TRAINER.FINETUNE = args.finetune

    if args.finetune_backbone:
        cfg.TRAINER.FINETUNE_BACKBONE = args.finetune_backbone

    if args.finetune_attn:
        cfg.TRAINER.FINETUNE_ATTN = args.finetune_attn

    if args.finetune_text:
        cfg.TRAINER.FINETUNE_TEXT = args.finetune_text

    if args.base_lr_mult:
        cfg.OPTIM.BASE_LR_MULT = args.base_lr_mult

    if args.backbone_lr_mult:
        cfg.OPTIM.BACKBONE_LR_MULT = args.backbone_lr_mult

    if args.text_lr_mult:
        cfg.OPTIM.TEXT_LR_MULT = args.text_lr_mult

    if args.attn_lr_mult:
        cfg.OPTIM.ATTN_LR_MULT = args.attn_lr_mult

    if args.max_epochs:
        cfg.OPTIM.MAX_EPOCH = args.max_epochs

    if args.warmup_epochs is not None:
        cfg.OPTIM.WARMUP_EPOCH = args.warmup_epochs


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg