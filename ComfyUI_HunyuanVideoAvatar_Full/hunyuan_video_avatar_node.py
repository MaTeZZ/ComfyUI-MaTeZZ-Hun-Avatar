import os
import torch
import numpy as np
from omegaconf import OmegaConf
import folder_paths # ComfyUI's way to get model paths

# Placeholder for HunyuanVideoSampler and tranformer_load
# These will be properly defined or imported in a later step.
# For now, we'll create dummy versions to allow the class structure to be defined.

def tranformer_load(args):
    print(f"Dummy tranformer_load called with args: {args.ckpt}")
    # In a real scenario, this would load and return the model
    # For now, returning a placeholder object
    class DummyModel:
        def __init__(self, args):
            self.args = args
            # Simulate text_encoder and text_encoder_2 attributes if HY_Avatar_PreData expects them
            self.text_encoder = None
            self.text_encoder_2 = None
    return DummyModel(args)

# Define the new model directory name
AVATAR_MODEL_DIR = "Avatar" # As per user request

# Ensure the new model directory exists in ComfyUI/models/
Hunyuan_Avatar_Weigths_Path = os.path.join(folder_paths.models_dir, AVATAR_MODEL_DIR)
if not os.path.exists(Hunyuan_Avatar_Weigths_Path):
    os.makedirs(Hunyuan_Avatar_Weigths_Path, exist_ok=True)

# Add our custom model folder to ComfyUI's list
folder_paths.add_model_folder_path(AVATAR_MODEL_DIR, Hunyuan_Avatar_Weigths_Path)

MAX_SEED = np.iinfo(np.int32).max

class HY_VIDEO_AVATAR_LOADER:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Get list of .pt files from the new 'Avatar' directory
        # Check if Hunyuan_Avatar_Weigths_Path exists before listing directory
        if not os.path.exists(Hunyuan_Avatar_Weigths_Path):
            os.makedirs(Hunyuan_Avatar_Weigths_Path, exist_ok=True)
        model_files = [f for f in os.listdir(Hunyuan_Avatar_Weigths_Path) if f.endswith(".pt")]
        return {
            "required": {
                "transformer_model": (["none"] + model_files,),
                "use_fp8": ("BOOLEAN", {"default": True}),
                "cpu_offload": ("BOOLEAN", {"default": True}),
                # Add other essential args from HY_Avatar_Loader in SM version if they are set at load time
            },
        }

    RETURN_TYPES = ("HY_MODEL", "HY_ARGS") # Custom types for clarity
    RETURN_NAMES = ("hunyuan_model", "hunyuan_args")
    FUNCTION = "load_model"
    CATEGORY = "HunyuanVideoAvatar_Full"

    def load_model(self, transformer_model, use_fp8, cpu_offload):
        if transformer_model == "none":
            raise FileNotFoundError(f"No transformer model selected. Please download models to ComfyUI/models/{AVATAR_MODEL_DIR}/ and select one.")

        model_path = folder_paths.get_full_path(AVATAR_MODEL_DIR, transformer_model)
        if not model_path or not os.path.exists(model_path):
            # Try to construct path if get_full_path returns None but file might exist under base models_dir
            if not model_path:
                model_path = os.path.join(Hunyuan_Avatar_Weigths_Path, transformer_model)
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model {transformer_model} not found. Expected at {Hunyuan_Avatar_Weigths_Path} or {folder_paths.get_folder_paths(AVATAR_MODEL_DIR)}. Please check ComfyUI/models/{AVATAR_MODEL_DIR}/.")


        # These arguments are based on the ComfyUI_HunyuanAvatar_Sm node
        # We will refine these and add more controls in subsequent steps
        vae_str="884-16c-hy0801" # Example, might need to be configurable
        vae_channels = int(vae_str.split("-")[1][:-1]) # Example

        args_dict = {
            "ckpt": model_path,
            "model": "HYVideo-T/2", # This might need to be inferred or an input
            "video_size": 512, # Default, make configurable later
            "load_key": "module", # "module" or "ema"
            "sample_n_frames": 129,
            "seed": 128, # Default, will be overridden by sampler
            "image_size": 704, # Default, make configurable later
            "cfg_scale": 7.5, # Default, make configurable later
            "ip_cfg_scale": 0, # Default, make configurable later
            "infer_steps": 50, # Default, make configurable later
            "use_deepcache": 1, # Default, make configurable later
            "flow_shift_eval_video": 5.0, # Default, make configurable later
            "use_linear_quadratic_schedule": True,
            "use_attention_mask": True,
            "linear_schedule_end": 25,
            "flow_solver": "euler",
            "flow_reverse": True,
            "save_path": folder_paths.get_output_directory(), # ComfyUI output path
            "use_fp8": use_fp8,
            "cpu_offload": cpu_offload,
            "infer_min": True, # Default, make configurable later
            "precision": "bf16",
            "reproduce": True,
            "num_images": 1,
            "val_disable_autocast": False,
            "pos_prompt": "", # Will come from predata node
            "neg_prompt": "", # Will come from predata node
            "save_path_suffix": "_ComfyUI",
            "pad_face_size": 0.7, # Default, make configurable later
            "item_name": "Hunyuan_Avatar_Full",
            # "use_deepcache": 1, # Duplicate, remove
            "latent_channels": vae_channels,
            "rope_theta": 256,
            "vae": vae_str, # Default, make configurable later
            "vae_tiling": True, # Default, make configurable later
            "vae_precision": "fp16",
            "text_encoder": "llava-llama-3-8b", # Default, make configurable later
            "tokenizer": "llava-llama-3-8b", # Default, make configurable later
            "text_encoder_precision": "fp16",
            "text_states_dim": 4096,
            "text_len": 256,
            "text_encoder_infer_mode": "encoder",
            "prompt_template_video": "li-dit-encode-video",
            "hidden_state_skip_layer": 2,
            "apply_final_norm": True,
            "text_encoder_2": "clipL", # Default, make configurable later
            "text_encoder_precision_2": "fp16",
            "text_states_dim_2": 768,
            "tokenizer_2": "clipL",
            "text_len_2": 77,
            "text_projection": "single_refiner",
            "daul_role": False, # Default, make configurable later (for multi-character)
            "face_size": 3.0, # Default, make configurable later (for FAA)
            # Parameters for AEM (Audio Emotion Module) will be added here later
            "emotion_reference_image_path": None, # Placeholder for emotion image
            "emotion_intensity": 1.0, # Placeholder
        }
        args = OmegaConf.create(args_dict)

        print(f"Attempting to load model from: {args.ckpt}")
        # This will eventually call the real HunyuanVideoSampler
        hunyuan_video_sampler = tranformer_load(args)

        # For now, hunyuan_video_sampler is a DummyModel.
        # In the real implementation, it would be the actual loaded model pipeline.
        # We pass the args along as they contain configuration needed by other nodes.
        return (hunyuan_video_sampler, args)

# Future nodes (PreData, Sampler) will be added below
