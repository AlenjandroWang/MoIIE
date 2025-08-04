import torch
import torch.nn as nn
import torch.nn.functional as F
from .eva_clip_processors import EvaClipImageTrainProcessor
from .eva_vit import EVAEncoderWrapper
from .factory import list_models, add_model_config, get_model_config



class EvaClipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.config = get_model_config(vision_tower)
        self._interp_size = 729


        if not delay_load:
            print(f"Loading EVA ViT: {self.vision_tower_name}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        else:
            self.cfg_only = self.config
    
    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features



    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.image_processor = EvaClipImageTrainProcessor(self.config["vision_cfg"]["image_size"])
        self.vision_tower = EVAEncoderWrapper(self.config)
        print(f"Loaded image processor: {self.image_processor}")
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True


    def forward(self, images,return_cls_token=False):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0)).to(image.dtype)
                image_features = self.interpolate(image_features)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).to(images.dtype)
            image_features = self.interpolate(image_features)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config["vision_cfg"]["width"]

    @property
    def num_patches(self):
        if self._interp_size is None:
            return (self.config["vision_cfg"]["image_size"] // self.config["vision_cfg"]["patch_size"]) ** 2
        else:
            return self._interp_size

    @property
    def num_patches_per_side(self):
        return self.config["vision_cfg"]["image_size"] // self.config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config["vision_cfg"]["image_size"]
