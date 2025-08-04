import os
from .eva_clip_vision.eva_clip_encoder import EvaClipVisionTower
from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
from .clip.clip_encoder import CLIPVisionTower

from .dino_encoder import DinoVisionTower
from .vae_encoder import  VAEVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)

    if 'clip' in vision_tower.lower():
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower.lower():
        if use_s2:
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    elif 'dino' in vision_tower.lower():
        return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'eva' in vision_tower.lower():
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    elif 'flux' in vision_tower.lower():
        return VAEVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_vision_tower_aux(vision_tower_cfg, **kwargs):
    vision_tower_aux = getattr(vision_tower_cfg, 'mm_vision_tower_aux', getattr(vision_tower_cfg, 'vision_tower_aux', None))
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)
    if 'clip-clip' in vision_tower_aux.lower():
        if use_s2:
            return CLIPVisionTowerS2(vision_tower_aux, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower_aux.lower():
        if use_s2:
            return SiglipVisionTowerS2(vision_tower_aux, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    
    elif 'dino' in vision_tower_aux.lower():
        return DinoVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    elif 'eva' in vision_tower_aux.lower():
        return EvaClipVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    elif 'flux' in vision_tower_aux.lower():
        return VAEVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower_aux}')