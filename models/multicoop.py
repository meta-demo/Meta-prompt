import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['multicoop', 'MultiCoop']


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, device_id):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.device_id = device_id

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    def obtain_label_emds(self, classnames):
        tokenized_names = torch.cat([clip.tokenize(name) for name in classnames])
        if torch.cuda.is_available():
                device = torch.device("cuda", self.device_id)
        else:
            device = torch.device("cpu")
        tokenized_names = tokenized_names.to(device)

        with torch.no_grad():
            
            embedding = self.token_embedding(tokenized_names).type(self.dtype)
            x = embedding + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            x = x.mean(dim=1)
            x = x @ self.text_projection
        # print("classnme embs:")
        # print(x.shape)

        return x

class MLP(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mid_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, embeddings):
        
        output = self.linear1(embeddings)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        
        return output


class MultiPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        
        # prompt length
        n_ctx = cfg.TRAINER.COOP_MLC.N_CTX
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        key_dim = clip_model.text_projection.shape[1]
        pool_size = cfg.TRAINER.COOP_MLC.POOL_SIZE
        prompt_key_init = cfg.TRAINER.COOP_MLC.PROMPT_KEY_INIT
     
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # initializing prompt key
        # prompt_key_init = 'uniform'
        # key_shape = (pool_size, ctx_dim)
        key_shape = (pool_size, key_dim)
        if prompt_key_init == 'zero':
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
        elif prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
        elif prompt_key_init == 'normal':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.normal_(self.prompt_key, std=0.02)
        # prompt_pool_shape = (pool_size, n_ctx, ctx_dim)
        print("Initializing a generic context")
        ctx_vectors = torch.empty(pool_size, n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Size of prompt pool: {pool_size}")
        print(f"Number of context words (tokens): {n_ctx}")

        self.prompt = nn.Parameter(ctx_vectors)  # to be optimized

        self.n_ctx = n_ctx
        self.class_token_position = 'end'
        self.token_embedding = clip_model.token_embedding
        self.device_id = cfg.TRAINER.DEVICEID

    def forward(self, classnames, name_embs):
        
        n_cls = len(classnames)
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        if torch.cuda.is_available():
            device = torch.device("cuda", self.device_id)
        else:
            device = torch.device("cpu")
        tokenized_prompts = tokenized_prompts.to(device)

        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)
        
        
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx :, :]
        
        prompt = self.prompt.to(device)
        prompt_key = self.prompt_key.to(device)

        name_embs = name_embs / name_embs.norm(dim=-1, keepdim=True)
        prompt_key = prompt_key / prompt_key.norm(dim=-1, keepdim=True)

        # calculate weights
        weights = name_embs @ prompt_key.T  # n_cls, pool_size
        weights = F.softmax(weights, dim=1)  # n_cls, pool_size
        prompt = prompt.permute(2, 0, 1)  # dim, pool_size, length
        weights = weights.unsqueeze(dim=0)  # 1, n_cls, pool_size
        weighted_prompts = torch.matmul(weights, prompt)  # dim, n_cls, length
        weighted_prompts = weighted_prompts.permute(1, 2, 0)   # n_cls, length, dim
        
        
        

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    weighted_prompts,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            raise ValueError

        return prompts, tokenized_prompts, weights



class MultiCoop(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        
        self.prompt_learner = MultiPromptLearner(cfg, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, cfg.TRAINER.DEVICEID)
        
        self.dtype = clip_model.dtype

        self.count_mlp = MLP(input_dim=1024, mid_dim=300, output_dim=4)
       

        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME
       

    def forward(self, classnames, image):
        # get image and text features
        image_features = self.image_encoder(image.type(self.dtype))
        name_embs = self.text_encoder.obtain_label_emds(classnames)
        prompts, tokenized_prompts, weights = self.prompt_learner(classnames, name_embs)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        print("image features: ", image_features.shape)
        print("text features: ", text_features.shape)

        count_outputs = self.count_mlp(image_features)
        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        


        return image_features, text_features, count_outputs, weights

    @property
    def network_name(self):
        name = ''
        name += 'DualCoop-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        print("Prompt params: ")
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                print(name)
                params.append(param)
        return params
    
    def mlpcount_params(self):
        params = []
        print("Count params: ")
        for name, param in self.named_parameters():
            if 'count_mlp' in name:
                print(name)
                params.append(param)
        return params


def multicoop(cfg, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building multicoop")
    model = MultiCoop(cfg, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda", cfg.TRAINER.DEVICEID)
    else:
        device = torch.device("cpu")
    model.to(device)

    
    return model
