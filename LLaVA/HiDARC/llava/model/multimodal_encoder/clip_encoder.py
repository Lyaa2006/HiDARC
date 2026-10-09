import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTextConfig
from ..spectral_pca import load_spectral_pca


def is_intern_vit_6b_model(vision_tower_name):
    model_names = ["intern_vit_6b", "internvit_6b", "InternViT-6B", "internvit6b"]
    return any(name in vision_tower_name for name in model_names)


def get_vision_feature_width(vision_tower_name):
    """Return the patch width consumed by HiDESC spectral routing."""
    if is_intern_vit_6b_model(vision_tower_name):
        from .intern_vit_6b.configuration_intern_vit import InternVisionConfig

        return int(InternVisionConfig.from_pretrained(vision_tower_name).hidden_size)
    return int(CLIPVisionConfig.from_pretrained(vision_tower_name).projection_dim)


def get_spectral_route_feature_width(
    vision_tower_name,
    spectral_pca_path=None,
    spectral_route_channel_dim=None,
):
    """Return the channel width used by the HiDESC spectral descriptor."""
    native_width = get_vision_feature_width(vision_tower_name)
    if not is_intern_vit_6b_model(vision_tower_name):
        if spectral_pca_path:
            raise ValueError("spectral_pca_path is only supported for InternViT")
        return native_width
    if spectral_pca_path is None:
        if spectral_route_channel_dim is not None:
            raise ValueError(
                "spectral_route_channel_dim was configured for InternViT, but "
                "spectral_pca_path is missing."
            )
        return native_width
    projector = load_spectral_pca(
        spectral_pca_path,
        expected_input_dim=native_width,
        expected_output_dim=spectral_route_channel_dim,
    )
    return projector.output_dim


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.spectral_pca_path = getattr(args, 'spectral_pca_path', None)
        self.spectral_route_channel_dim = getattr(args, 'spectral_route_channel_dim', None)
        self.spectral_pca = None
        self.configure_spectral_pca(self.spectral_pca_path, self.spectral_route_channel_dim)

        if not delay_load:
            self.load_model()
        else:
            if is_intern_vit_6b_model(self.vision_tower_name):
                from .intern_vit_6b.configuration_intern_vit import InternVisionConfig

                self.cfg_only = InternVisionConfig.from_pretrained(self.vision_tower_name)
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if is_intern_vit_6b_model(self.vision_tower_name):
            # InternViT-6B checkpoints use the same 336px preprocessing path
            # as the original InternVL LLaVA integration.
            crop_size = 448 if "448" in self.vision_tower_name else 336
            self.image_processor = CLIPImageProcessor(
                crop_size=crop_size,
                do_center_crop=True,
                do_normalize=True,
                do_resize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                size=crop_size,
            )
            from .intern_vit_6b.modeling_intern_vit import InternVisionModel

            self.vision_tower = InternVisionModel.from_pretrained(self.vision_tower_name)
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def configure_spectral_pca(self, path=None, output_dim=None):
        self.spectral_pca_path = path
        self.spectral_route_channel_dim = output_dim
        self.spectral_pca = None
        if path:
            if not is_intern_vit_6b_model(self.vision_tower_name):
                raise ValueError("spectral_pca_path is only supported for InternViT")
            native_width = get_vision_feature_width(self.vision_tower_name)
            self.spectral_pca = load_spectral_pca(
                path,
                expected_input_dim=native_width,
                expected_output_dim=output_dim,
            )
            self.spectral_pca.requires_grad_(False)
        elif is_intern_vit_6b_model(self.vision_tower_name) and output_dim is not None:
            raise ValueError("InternViT spectral_route_channel_dim requires spectral_pca_path")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            images = torch.cat(
                [
                    image.unsqueeze(0) if image.ndim == 3 else image
                    for image in images
                ],
                dim=0,
            )

        input_dtype = images.dtype
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        selected_patch_features = self.feature_select(image_forward_outs).to(input_dtype)

        final_patch_features = image_forward_outs.last_hidden_state[:, 1:].to(input_dtype)
        if is_intern_vit_6b_model(self.vision_tower_name):
            # InternViT has no CLIP visual projection. Keep the native patch
            # representation for HiDESC's optional spectral descriptor.
            projected_patch_features = final_patch_features
            if self.spectral_pca is not None:
                projected_patch_features = self.spectral_pca(projected_patch_features)
            clip_image_features = image_forward_outs.pooler_output.to(input_dtype)
        else:
            # `last_hidden_state` is the final vision-layer output. The
            # projection is the same CLIP projection used for image_embeds.
            projected_patch_features = self.vision_tower.visual_projection(
                final_patch_features.to(self.vision_tower.visual_projection.weight.dtype)
            ).to(input_dtype)
            clip_image_features = image_forward_outs.image_embeds.to(input_dtype)

        return (
            clip_image_features,
            selected_patch_features,
            final_patch_features,
            projected_patch_features,
        )

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPTextTower(nn.Module):
    def __init__(self, text_tower, args, delay_load=False):
        super().__init__()
        
        self.is_loaded = False

        self.text_tower_name = text_tower
        self.select_layer = args.mm_text_select_layer
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPTextConfig.from_pretrained(self.text_tower_name)

    def load_model(self):
        self.text_tower = CLIPTextModel.from_pretrained(self.text_tower_name)
        self.text_tower.requires_grad_(False)

        self.is_loaded = True


    @torch.no_grad()
    def forward(self, text_inputs, return_hidden_states=False):
        # if type(texts_input_ids) is list:
        #     text_features = []
        #     for text_input_ids in texts_input_ids:
        #         text_forward_out = self.text_tower(text_input_ids.to(self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
        #         text_feature = self.feature_select(text_forward_out).to(text_input_ids.dtype)
        #         text_features.append(text_feature)
        # else:
        text_forward_outs = self.text_tower(**(text_inputs.to(self.device)))
        if return_hidden_states:
            text_hidden_features = text_forward_outs.last_hidden_state.to(self.dtype)
            text_features = text_forward_outs.pooler_output.to(self.dtype)

            return [text_hidden_features, text_features]
        else:
            text_features = text_forward_outs.pooler_output.to(self.dtype)
            return text_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.text_tower.dtype

    @property
    def device(self):
        return self.text_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.text_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
