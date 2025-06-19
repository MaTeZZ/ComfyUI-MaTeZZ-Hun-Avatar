# ComfyUI_HunyuanVideoAvatar_Full
# This file is used to register the custom nodes with ComfyUI.

from .hunyuan_video_avatar_node import HY_VIDEO_AVATAR_LOADER
# Import other nodes when they are defined:
# from .hunyuan_video_avatar_node import HY_VIDEO_AVATAR_PREDATA, HY_VIDEO_AVATAR_SAMPLER

NODE_CLASS_MAPPINGS = {
    "HY_VideoAvatar_Loader_Full": HY_VIDEO_AVATAR_LOADER,
    # "HY_VideoAvatar_PreData_Full": HY_VIDEO_AVATAR_PREDATA,
    # "HY_VideoAvatar_Sampler_Full": HY_VIDEO_AVATAR_SAMPLER,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HY_VideoAvatar_Loader_Full": "Hunyuan Video Avatar Loader (Full)",
    # "HY_VideoAvatar_PreData_Full": "Hunyuan Video Avatar PreData (Full)",
    # "HY_VideoAvatar_Sampler_Full": "Hunyuan Video Avatar Sampler (Full)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("ComfyUI_HunyuanVideoAvatar_Full custom node loaded with Loader.")
