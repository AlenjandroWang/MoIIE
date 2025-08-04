import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor, Dinov2Config







class DinoVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.vision_tower_name = vision_tower
        self._image_size = 518
        self._interp_size = 729
        self._patch_size = 14  # default patch size
        self.select_feature = 'patch'
        self.is_loaded = False

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = Dinov2Config.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):

        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name)
        """ValueError: Dinov2Model does not support `device_map='auto'`. To implement support, the model class needs to implement the `_no_split_modules` attribute."""
        self.vision_tower._no_split_modules = ["Dinov2SwiGLUFFN"]

        _image_size = self.vision_tower.config.image_size
        if self._image_size is None:
            self._image_size = _image_size
        else:
            print(f"Overriding DinoVisionTower image size of {_image_size} with {self._image_size}")


        shortest_edge = self._image_size

        processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, crop_size=dict(height=self._image_size, width=self._image_size), size=dict(shortest_edge=shortest_edge))

        print(f"Dino Vision Processor: {processor}")
        self.image_processor = processor

        # Assign the output channels of the projection convolution as the hidden size
        self._hidden_size = self.vision_tower.embeddings.patch_embeddings.projection.out_channels
        # Assign the first value of the stride of the projection convolution as the patch size
        self._patch_size = self.vision_tower.embeddings.patch_embeddings.projection.stride[0]

        #print(self._hidden_size, self._patch_size)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs["last_hidden_state"]  # batch_size, sequence_length, hidden_size

        if self.select_feature == 'cls_patch':
            image_features = sequence_output
        elif self.select_feature == 'patch':
            image_features = sequence_output[:, 1:]
        elif self.select_feature == 'cls':
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

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

    def forward(self, images, return_cls_token=False):
        # logger.warning(f"images shape: {images.shape}")
        with torch.no_grad():
            image_forward_outs = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))
            # logger.warning(f"image_forward_outs shape: {image_forward_outs['last_hidden_state'].shape}")
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # logger.warning(f"image_features shape: {image_features.shape}")
            interp_features = self.interpolate(image_features)
            # logger.warning(f"interp_features shape: {interp_features.shape}")
            return interp_features

    @property
    def num_patches_per_side(self):
        return int(self.num_patches ** 0.5)
    
    @property
    def hidden_size(self):
        return self._hidden_size
    
    @property
    def dtype(self):
        return self.vision_tower.dtype
    
    @property
    def device(self):
        return self.vision_tower.device

    @property
    def num_patches(self):
        if self._interp_size is None:
            return (self._image_size // self._patch_size) ** 2
        else:
            return self._interp_size