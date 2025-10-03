#!/usr/bin/env python3
"""
LLaVA-OneVision 3D Drone Navigation Fine-tuning Script.
Based on official LLaVA training framework with 3D action space support.
"""

import os
import sys
import copy
import json
import pathlib
from typing import Dict, Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image

sys.path.append('.')
sys.path.append('./LLaVA-NeXT')

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_anyres_image, tokenizer_image_token
from llava.utils import rank0_print

from llava.train.train import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    make_supervised_data_module,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    find_all_linear_names,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
    maybe_zero_3
)


def create_deepspeed_config_4gpu():
    """Create DeepSpeed configuration optimized for 4 GPUs"""
    config = {
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 200000000,
            "stage3_prefetch_bucket_size": 20000000,
            "stage3_param_persistence_threshold": 400000,
            "stage3_max_live_parameters": 400000000,
            "stage3_max_reuse_distance": 400000000,
            "stage3_gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
        "communication_data_type": "bf16",
        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": None
        }
    }

    with open("deepspeed_config_drone_3d_4gpu.json", 'w') as f:
        json.dump(config, f, indent=2)

    return config


def main():
    """Main training function - 4 GPU optimized for 3D drone navigation"""

    # Environment setup for 4 GPUs
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '9997')

    # NCCL optimization for 4 GPUs
    os.environ.setdefault('NCCL_TREE_THRESHOLD', '0')
    os.environ.setdefault('NCCL_P2P_DISABLE', '0')
    os.environ.setdefault('NCCL_IB_DISABLE', '0')
    os.environ.setdefault('NCCL_DEBUG', 'INFO')

    # Create DeepSpeed config
    create_deepspeed_config_4gpu()

    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("Starting 3D Drone Navigation Fine-tuning with 4GPU Configuration...")
    rank0_print(f"Available GPUs: {torch.cuda.device_count()}")
    rank0_print(f"3D Drone data: {data_args.data_path}")
    rank0_print(f"Image folder: {data_args.image_folder}")
    rank0_print(f"Output directory: {training_args.output_dir}")

    # Set conversation template
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]

    rank0_print(f"Using conversation template: {model_args.version}")

    # Compute dtype
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # BitsAndBytes configuration
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    # Load model
    rank0_print("Loading LLaVA-OneVision model...")
    model = LlavaQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **bnb_model_from_pretrained_args,
    )

    model.config.use_cache = False

    # Quantization preparation
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # Gradient checkpointing
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load tokenizer
    rank0_print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )

    # Set pad token
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    # Initialize vision modules
    rank0_print("Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    # Configure image processing parameters
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position = model_args.mm_newline_position

    # Parse grid pinpoints
    if data_args.image_grid_pinpoints is not None:
        if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
            import re
            import ast
            try:
                patch_size = data_args.image_processor.size["shortest_edge"]
            except:
                patch_size = data_args.image_processor.size[0]

            matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
            range_start = tuple(map(int, matches[0]))
            range_end = tuple(map(int, matches[-1]))
            grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in
                              range(range_start[1], range_end[1] + 1)]
            data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
        elif isinstance(data_args.image_grid_pinpoints, str):
            data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

    # Configure trainable parts
    if model_args.mm_tunable_parts is not None:
        rank0_print(f"Configuring tunable parts: {model_args.mm_tunable_parts}")
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        if hasattr(model.get_model(), 'mm_projector'):
            model.get_model().mm_projector.requires_grad_(False)
        if hasattr(model.get_model(), 'vision_resampler'):
            model.get_model().vision_resampler.requires_grad_(False)

        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_mlp_adapter" in tunable_parts and hasattr(model.get_model(), 'mm_projector'):
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if "mm_vision_resampler" in tunable_parts and hasattr(model.get_model(), 'vision_resampler'):
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True
        if "mm_vision_tower" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad_(True)
        if "mm_language_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                    param.requires_grad_(True)
    else:
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
        if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
            model.requires_grad_(False)
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if model_args.tune_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
        if training_args.freeze_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
        if model_args.unfreeze_mm_vision_tower:
            vision_tower.requires_grad_(True)
        else:
            vision_tower.requires_grad_(False)

    # LoRA configuration
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params / 1e6:.2f}M")
    rank0_print(f"Trainable parameters: ~{trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")

    # Set multimodal attributes
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr

    # Handle quantized model special cases
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    # Initialize vision tokenizer
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Handle quantized model layer precision
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Create standard LLaVA data module
    rank0_print("Creating standard LLaVA data module...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 3D drone dataset statistics
    rank0_print("3D Drone dataset statistics:")
    if hasattr(data_module, 'train_dataset') and data_module['train_dataset'] is not None:
        train_size = len(data_module['train_dataset'])
        rank0_print(f"  Training samples: {train_size:,}")

        try:
            with open(data_args.data_path, 'r') as f:
                data = json.load(f)

            # 3D action space mapping
            drone_3d_actions = {
                "START": "Start state",
                "MOVE_FORWARD": "Move forward 25cm", 
                "TURN_LEFT": "Turn left 15 degrees",
                "TURN_RIGHT": "Turn right 15 degrees",
                "MOVE_UP": "Move up 20cm",
                "MOVE_DOWN": "Move down 20cm",
                "STOP": "Stop"
            }

            action_counts = {}
            for item in data:
                if 'conversations' in item and len(item['conversations']) >= 2:
                    action = item['conversations'][1]['value']
                    action_counts[action] = action_counts.get(action, 0) + 1

            rank0_print('  3D action space distribution:')
            for action, count in sorted(action_counts.items()):
                percentage = count / len(data) * 100
                action_desc = drone_3d_actions.get(action, "Unknown action")
                action_type = "3D Extended" if action in ["MOVE_UP", "MOVE_DOWN"] else "Original 2D"
                rank0_print(f'    {action}: {count:,} ({percentage:.1f}%) - {action_desc} ({action_type})')

            # 3D-specific action statistics
            drone_3d_specific = sum(action_counts.get(action, 0) for action in ["MOVE_UP", "MOVE_DOWN"])
            if drone_3d_specific > 0:
                rank0_print(f'  3D-specific actions total: {drone_3d_specific:,} ({drone_3d_specific / len(data) * 100:.1f}%)')

        except Exception as e:
            rank0_print(f"  Unable to compute 3D action distribution: {e}")

    # Create trainer
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # Start training
    rank0_print("Starting 3D drone navigation training (using standard LLaVA framework)...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    # Save model
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"3D drone navigation model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
