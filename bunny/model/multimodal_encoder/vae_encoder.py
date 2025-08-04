import torch.nn as nn
from diffusers import AutoencoderKL
import torch.nn.functional as F
import torch
from einops import rearrange



class VAEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
     

        if not delay_load:
            self.load_model()
        # elif self.unfreeze:
        #     self.load_model()
        else:
            self.vision_tower = AutoencoderKL.from_pretrained(self.vision_tower_name)
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        

        self.vision_tower = AutoencoderKL.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
    
    def forward(self, images, return_cls_token=False):
        with torch.no_grad():
            posterior = self.encode(images.to(device=self.device, dtype=self.dtype)).latent_dist
            z_q = (posterior.sample() - self.shift_factor) * self.scaling_factor
            # group each (2x2) window
            z_q = z_q.unfold(2, 2, 2).unfold(3, 2, 2)
            z_q = rearrange(z_q, 'b c h w p1 p2 -> b (h w) (c p1 p2)').contiguous()
            return z_q
    

    @property
    def scaling_factor(self):
        return self.vision_tower.config.scaling_factor

    @property
    def shift_factor(self):
        return self.vision_tower.config.shift_factor

    @property
    def hidden_size(self):
        return self.vision_tower.config.latent_channels * 4

    def encode(self, x):
        return self.vision_tower.encode(x)

    def decode(self, z):
        return self.vision_tower.decode(z)
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
        
