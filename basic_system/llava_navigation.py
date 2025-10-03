#!/usr/bin/env python3
"""
LLaVA-OneVision Navigation Model
Supports dual-target navigation with visible guide landmarks and actual target landmarks.
Optimized for low-latency inference with visualization capabilities.
"""

import torch
import warnings
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from datetime import datetime
import time

warnings.filterwarnings('ignore')

from utils import get_best_gpu, set_gpu_device, get_safe_output_dir

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates


class LLaVANavigationModel:
    """LLaVA-OneVision navigation model with dual-target support and inference visualization."""

    def __init__(self, model_path: str, preferred_gpu_id: int = None, quantization_level: int = 0):
        """Initialize LLaVA-OneVision model."""
        print("Initializing LLaVA-OneVision Navigation Model...")
        self.quantization_level = quantization_level

        if preferred_gpu_id is None:
            gpu_id, gpu_name, gpu_memory = get_best_gpu()
        else:
            gpu_id = preferred_gpu_id

        self.device = set_gpu_device(gpu_id)

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self._load_llava_model()

        # Inference configuration
        self.temperature = 1.2
        self.top_p = 0.9
        self.max_new_tokens = 16
        self.do_sample = True

        self.navigation_params = {
            'temperature': 1.2,
            'top_p': 0.9,
            'max_new_tokens': 16,
            'do_sample': True
        }

        self.scene_analysis_params = {
            'temperature': None,
            'top_p': None,
            'max_new_tokens': 32,
            'do_sample': False
        }

        self.instruction_parsing_params = {
            'temperature': 0.1,
            'top_p': 0.9,
            'max_new_tokens': 128,
            'do_sample': True,
            'num_beams': 1,
        }

        self.topological_graph = None
        self.enable_visualization = True
        self.visualization_dir = None
        self.inference_history = []

        print("LLaVA-OneVision Navigation Model initialized")

    def set_topological_graph(self, topological_graph):
        """Set topological graph reference."""
        self.topological_graph = topological_graph

    def set_visualization_config(self, enable: bool = True, output_dir: str = None):
        """Configure visualization settings."""
        self.enable_visualization = enable
        if output_dir is None:
            self.visualization_dir = get_safe_output_dir() / "inference_visualization"
        else:
            self.visualization_dir = Path(output_dir)

        if self.enable_visualization:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)

    def _load_llava_model(self):
        """Load LLaVA-OneVision model components (supports both LoRA and standard models)."""
        try:
            is_lora_model = self._check_if_lora_model()

            if is_lora_model:
                print("Detected LoRA model, loading with LoRA adapter...")
                self._load_lora_model()
            else:
                print("Detected standard model, loading with standard method...")
                self._load_standard_model()

            if hasattr(self.model.config, "_attn_implementation") and \
               self.model.config._attn_implementation == "flash_attention_2":
                print("Flash Attention 2 is active.")

            self.model.eval()

            sample_param = next(self.model.parameters())
            self.model_dtype = sample_param.dtype

            self.vision_tower = self.model.get_vision_tower()
            if not self.vision_tower.is_loaded:
                self.vision_tower.load_model()
            self.vision_tower = self.vision_tower.to(device=self.device, dtype=self.model_dtype)

        except Exception as e:
            print(f"Failed to load LLaVA model: {e}")
            raise e

    def _check_if_lora_model(self) -> bool:
        """Check if model is a LoRA model."""
        lora_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]

        for lora_file in lora_files:
            if (self.model_path / lora_file).exists():
                return True

        if (self.model_path / "pytorch_model.bin").exists():
            return False

        if (self.model_path / "config.json").exists():
            try:
                with open(self.model_path / "config.json", 'r') as f:
                    config = json.load(f)
                return "peft_type" in config or "target_modules" in config
            except:
                pass

        return False

    def _load_lora_model(self):
        """Load LoRA model."""
        from peft import PeftModel

        base_model_path = "/root/autodl-tmp/model/transformers/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"

        adapter_config_path = self.model_path / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                if "base_model_name_or_path" in adapter_config:
                    base_model_path = adapter_config["base_model_name_or_path"]
            except:
                pass

        print(f"Base model path: {base_model_path}")

        quantization_config = {}
        if self.quantization_level == 8:
            quantization_config['load_in_8bit'] = True
        elif self.quantization_level == 4:
            quantization_config['load_in_4bit'] = True

        self.tokenizer, base_model, self.image_processor, self.max_length = load_pretrained_model(
            model_path=base_model_path,
            model_base=None,
            model_name="llava_qwen",
            device_map={"": self.device},
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            **quantization_config
        )

        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.model_path),
            device_map={"": self.device}
        )

        print("LoRA model loaded successfully")

    def _load_standard_model(self):
        """Load standard model."""
        quantization_config = {}
        if self.quantization_level == 8:
            quantization_config['load_in_8bit'] = True
        elif self.quantization_level == 4:
            quantization_config['load_in_4bit'] = True

        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path=str(self.model_path),
            model_base=None,
            model_name="llava_qwen",
            device_map={"": self.device},
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            **quantization_config
        )

    def parse_navigation_instruction(self, instruction: str) -> List[str]:
        """Parse navigation instruction into landmark sequence."""
        try:
            mixed_targets = self._extract_mixed_targets(instruction)

            if not mixed_targets:
                return [instruction]

            landmark_sequence = self._convert_to_landmark_sequence(mixed_targets, instruction)

            return landmark_sequence if landmark_sequence else [instruction]

        except Exception as e:
            print(f"Mixed target parsing failed: {e}")
            return [instruction]

    def _extract_mixed_targets(self, instruction: str) -> List[Dict]:
        """Extract mixed target sequence (scenes + landmarks) from instruction."""
        try:
            parsing_params = self.instruction_parsing_params
            parsing_prompt = f"""Extract navigation targets from the instruction in the order they appear step by step:

"{instruction}"

Identify each target as either:
- "landmark": specific objects (door, table, chair, shelve, sofa, etc.)
- "scene": rooms/areas (kitchen, bedroom, hallway, etc.)

IMPORTANT: For landmarks, keep a complete description, including all adjectives, shapes, and identifying features (if any).

Examples:
- "the door" → keep as "door"
- "a square red table near the window" → keep as "a square red table"
- "kitchen" → simply keep as scene name "kitchen"

Return ONLY a valid JSON array like this example:
[{{"type": "scene", "name": "kitchen"}}, {{"type": "landmark", "name": "picture that has white background"}}]

JSON array:"""

            conv = self.prepare_conversation_template()
            if conv is None:
                return []

            conv.append_message(conv.roles[0], parsing_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=parsing_params['do_sample'],
                    temperature=parsing_params['temperature'] if parsing_params['do_sample'] else None,
                    top_p=parsing_params['top_p'] if parsing_params['do_sample'] else None,
                    max_new_tokens=parsing_params['max_new_tokens'],
                    num_beams=parsing_params.get('num_beams', 1),
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            mixed_targets = self._parse_mixed_targets_output(full_output)

            return mixed_targets

        except Exception as e:
            print(f"Mixed target extraction failed: {e}")
            return []

    def _parse_mixed_targets_output(self, full_output: str) -> List[Dict]:
        """Parse mixed targets from model output."""
        try:
            clean_output = full_output.replace("<|im_end|>", "").strip()

            json_pattern = r'\[.*?\]'
            json_match = re.search(json_pattern, clean_output, re.DOTALL)

            if json_match:
                try:
                    targets_data = json.loads(json_match.group(0))

                    if isinstance(targets_data, list):
                        mixed_targets = []
                        for target in targets_data:
                            if isinstance(target, dict) and 'type' in target and 'name' in target:
                                target_type = target['type'].lower()
                                target_name = str(target['name']).strip()
                                confidence = float(target.get('confidence', 0.8))

                                if target_type in ['landmark', 'scene'] and target_name and len(target_name) < 50:
                                    mixed_targets.append({
                                        'type': target_type,
                                        'name': target_name,
                                        'confidence': confidence
                                    })

                        return mixed_targets
                except json.JSONDecodeError:
                    pass
            
            print("JSON parsing failed, using fallback method.")
            return self._fallback_mixed_parsing(clean_output)

        except Exception:
            return []

    def _fallback_mixed_parsing(self, text: str) -> List[Dict]:
        """Fallback parsing method using keywords."""
        try:
            scene_keywords = [
                'kitchen', 'bedroom', 'living room', 'bathroom', 'garage', 'garden',
                'corridor', 'hallway', 'office', 'dining room', 'balcony', 'basement'
            ]

            words = text.lower().split()
            targets = []

            for word in words:
                word_clean = re.sub(r'[^\w\s]', '', word).strip()
                if len(word_clean) > 1:
                    is_scene = any(scene_kw in word_clean for scene_kw in scene_keywords)
                    targets.append({
                        'type': 'scene' if is_scene else 'landmark',
                        'name': word_clean,
                        'confidence': 0.7
                    })

            return targets[:5]

        except Exception:
            return []

    def _convert_to_landmark_sequence(self, mixed_targets: List[Dict], original_instruction: str) -> List[str]:
        """Convert mixed target sequence to pure landmark sequence."""
        if not self.topological_graph:
            return [target['name'] for target in mixed_targets if target['type'] == 'landmark']

        try:
            landmark_sequence = []
            previous_landmark_id = None

            for i, target in enumerate(mixed_targets):
                if target['type'] == 'landmark':
                    landmark_sequence.append(target['name'])

                    landmark_results = self.topological_graph.query_landmarks_by_language(
                        target['name'], top_k=1
                    )
                    if landmark_results:
                        previous_landmark_id = landmark_results[0][0]

                elif target['type'] == 'scene':
                    selected_landmark = self._select_scene_entry_landmark(
                        target['name'], i, previous_landmark_id, landmark_sequence
                    )

                    if selected_landmark:
                        landmark_sequence.append(selected_landmark)

                        landmark_results = self.topological_graph.query_landmarks_by_language(
                            selected_landmark, top_k=1
                        )
                        if landmark_results:
                            previous_landmark_id = landmark_results[0][0]

            return landmark_sequence if landmark_sequence else [original_instruction]

        except Exception as e:
            print(f"Sequence conversion failed: {e}")
            return [target['name'] for target in mixed_targets if target['type'] == 'landmark']

    def _select_scene_entry_landmark(self, scene_name: str, position: int,
                                     previous_landmark_id: Optional[int],
                                     current_sequence: List[str]) -> Optional[str]:
        """Select best entry landmark for scene."""
        try:
            scene_landmarks = self.topological_graph.get_landmarks_by_scene_type(scene_name)

            if not scene_landmarks:
                print(f"No landmarks found for scene: {scene_name}")
                return None

            if position == 0:
                selected_landmark_id = self.topological_graph.find_closest_scene_entry_to_current(
                    scene_landmarks
                )
            else:
                if previous_landmark_id is not None:
                    selected_landmark_id = self.topological_graph.find_closest_scene_entry_to_landmark(
                        scene_landmarks, previous_landmark_id
                    )
                else:
                    selected_landmark_id = self.topological_graph.find_closest_scene_entry_to_current(
                        scene_landmarks
                    )

            if selected_landmark_id is not None:
                landmark_name = self.topological_graph.get_landmark_representative_name(selected_landmark_id)
                return landmark_name if landmark_name else f"landmark_{selected_landmark_id}"

            return None

        except Exception as e:
            print(f"Scene entry selection failed for {scene_name}: {e}")
            return None

    def prepare_conversation_template(self):
        """Prepare conversation template."""
        try:
            conv_template = conv_templates["qwen_1_5"].copy()
            conv_template.messages = []
            return conv_template
        except:
            return None

    def process_image_for_model(self, image_path_or_pil):
        """Process image for model input (supports path string or PIL image)."""
        try:
            if isinstance(image_path_or_pil, str):
                if not Path(image_path_or_pil).exists():
                    return None, None
                image = Image.open(image_path_or_pil).convert('RGB')
            else:
                image = image_path_or_pil
                if image.mode != 'RGB':
                    image = image.convert('RGB')

            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_size = image.size

            if isinstance(image_tensor, list):
                image_tensor = [img.to(device=self.device, dtype=self.model_dtype) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(device=self.device, dtype=self.model_dtype)

            return image_tensor, image_size

        except Exception:
            return None, None

    def generate_dual_target_navigation_command(self, full_instruction: str, landmark_sequence: List[str],
                                                current_target: str, current_image_path: str,
                                                visible_landmark_id: int, target_landmark_id: int) -> dict:
        """Generate dual-target navigation command: visible guide landmark + true target landmark."""

        inference_record = {
            'timestamp': datetime.now().isoformat(),
            'input': {
                'full_instruction': full_instruction,
                'landmark_sequence': landmark_sequence,
                'current_target': current_target,
                'current_image_path': current_image_path,
                'visible_landmark_id': visible_landmark_id,
                'target_landmark_id': target_landmark_id
            },
            'processing': {},
            'output': {},
            'timing': {},
            'errors': []
        }

        start_time = time.time()

        try:
            if not self.topological_graph:
                result = {"action": "HOVER", "reason": "No graph", "confidence": 0.0}
                inference_record['errors'].append("No topological graph available")
                self._save_inference_record(inference_record, result)
                return result

            if visible_landmark_id is not None:
                visible_reference_path = self.topological_graph.get_landmark_reference_image(visible_landmark_id)
            else:
                visible_reference_path = None
            target_reference_path = self.topological_graph.get_landmark_reference_image(target_landmark_id)

            if not target_reference_path:
                result = {"action": "HOVER", "reason": "No reference images", "confidence": 0.0}
                inference_record['errors'].append(
                    f"Missing reference images: visible={visible_reference_path is not None}, target={target_reference_path is not None}")
                self._save_inference_record(inference_record, result)
                return result

            inference_record['processing']['visible_reference_path'] = visible_reference_path
            inference_record['processing']['target_reference_path'] = target_reference_path

            image_start = time.time()
            current_image, current_size = self.process_image_for_model(current_image_path)
            target_image, target_size = self.process_image_for_model(target_reference_path)
            
            if visible_reference_path:
                visible_image, visible_size = self.process_image_for_model(visible_reference_path)
            else:
                black_image = self._create_black_image()
                visible_image, visible_size = self.process_image_for_model(black_image)
                inference_record['processing']['using_black_image_for_visible'] = True
            
            image_processing_time = time.time() - image_start
            inference_record['timing']['image_processing_time'] = image_processing_time

            if current_image is None or visible_image is None or target_image is None:
                result = {"action": "HOVER", "reason": "Image processing failed", "confidence": 0.0}
                inference_record['errors'].append("Image processing failed")
                self._save_inference_record(inference_record, result)
                return result

            navigation_prompt = f"""What action should the robot take to navigate to target via visible guide:
Current observation: {DEFAULT_IMAGE_TOKEN}
Visible guide: {DEFAULT_IMAGE_TOKEN}
Target destination: {DEFAULT_IMAGE_TOKEN}
Answer with exactly one word: MOVE_FORWARD_3m, MOVE_FORWARD_6m, MOVE_FORWARD_9m, TURN_LEFT, TURN_RIGHT, or STOP"""

            inference_record['input']['navigation_prompt'] = navigation_prompt
            inference_record['input']['image_info'] = {
                'current_image_size': current_size,
                'visible_image_size': visible_size,
                'target_image_size': target_size
            }

            conv = self.prepare_conversation_template()
            if conv is None:
                result = {"action": "HOVER", "reason": "No template", "confidence": 0.0}
                inference_record['errors'].append("Failed to prepare conversation template")
                self._save_inference_record(inference_record, result)
                return result

            conv.append_message(conv.roles[0], navigation_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inference_record['processing']['full_prompt'] = prompt

            tokenize_start = time.time()
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            tokenize_time = time.time() - tokenize_start

            inference_record['timing']['tokenize_time'] = tokenize_time
            inference_record['processing']['input_token_count'] = input_ids.shape[1]

            parsing_params = self.navigation_params

            generation_start = time.time()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=[current_image, visible_image, target_image],
                    image_sizes=[current_size, visible_size, target_size],
                    do_sample=parsing_params['do_sample'],
                    temperature=parsing_params['temperature'] if parsing_params['do_sample'] else None,
                    top_p=parsing_params['top_p'] if parsing_params['do_sample'] else None,
                    max_new_tokens=parsing_params['max_new_tokens'],
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generation_time = time.time() - generation_start

            inference_record['timing']['generation_time'] = generation_time

            decode_start = time.time()
            full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            decode_time = time.time() - decode_start

            inference_record['timing']['decode_time'] = decode_time
            inference_record['output']['raw_model_output'] = full_output
            inference_record['processing']['output_token_count'] = output_ids.shape[1]

            command_start = time.time()
            navigation_command = self.extract_navigation_command(full_output)
            command_time = time.time() - command_start

            inference_record['timing']['command_extraction_time'] = command_time
            inference_record['output']['final_command'] = navigation_command

            total_time = time.time() - start_time
            inference_record['timing']['total_inference_time'] = total_time

            if self.enable_visualization:
                viz_start = time.time()
                self._visualize_dual_target_inference(inference_record, current_image_path,
                                                      visible_reference_path, target_reference_path)
                viz_time = time.time() - viz_start
                inference_record['timing']['visualization_time'] = viz_time

            self._save_inference_record(inference_record, navigation_command)

            return navigation_command

        except Exception as e:
            inference_record['errors'].append(f"Generation error: {str(e)}")
            result = {"action": "HOVER", "reason": f"Error: {str(e)[:20]}", "confidence": 0.0}
            self._save_inference_record(inference_record, result)
            return result

    def _create_black_image(self, target_size=(224, 224)):
        """Create black placeholder image for missing visible landmark."""
        black_array = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        black_image = Image.fromarray(black_array)
        return black_image

    def _visualize_dual_target_inference(self, inference_record: dict, current_image_path: str,
                                         visible_reference_path: str, target_reference_path: str):
        """Visualize dual-target inference process."""
        try:
            if not self.enable_visualization or not self.visualization_dir:
                return

            fig = plt.figure(figsize=(24, 16))

            timestamp = inference_record['timestamp']
            fig.suptitle(f'Dual-Target Navigation Inference\n{timestamp}',
                         fontsize=18, fontweight='bold', y=0.98)

            ax1 = plt.subplot2grid((3, 4), (0, 0))
            ax2 = plt.subplot2grid((3, 4), (0, 1))
            ax3 = plt.subplot2grid((3, 4), (0, 2))
            ax4 = plt.subplot2grid((3, 4), (0, 3))
            ax5 = plt.subplot2grid((3, 4), (1, 0))
            ax6 = plt.subplot2grid((3, 4), (1, 1))
            ax7 = plt.subplot2grid((3, 4), (1, 2))
            ax8 = plt.subplot2grid((3, 4), (1, 3))
            ax9 = plt.subplot2grid((3, 4), (2, 0), colspan=4)

            # Current observation
            try:
                current_img = cv2.imread(current_image_path)
                if current_img is not None:
                    current_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                    ax1.imshow(current_rgb)
                    h, w = current_rgb.shape[:2]
                    rect = Rectangle((0, 0), w - 1, h - 1, linewidth=4, edgecolor='blue', facecolor='none', alpha=0.8)
                    ax1.add_patch(rect)
                    ax1.text(10, 30, 'CURRENT VIEW',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.8),
                             fontsize=12, fontweight='bold', color='white')
                else:
                    ax1.text(0.5, 0.5, 'Failed to load\ncurrent image', ha='center', va='center',
                             transform=ax1.transAxes)
            except Exception:
                ax1.text(0.5, 0.5, f'Error loading\ncurrent image', ha='center', va='center',
                         transform=ax1.transAxes)

            ax1.set_title(f'Current Observation\n{Path(current_image_path).name}', fontsize=12, fontweight='bold')
            ax1.axis('off')

            # Visible guide landmark
            try:
                visible_img = cv2.imread(visible_reference_path)
                if visible_img is not None:
                    visible_rgb = cv2.cvtColor(visible_img, cv2.COLOR_BGR2RGB)
                    ax2.imshow(visible_rgb)
                    h, w = visible_rgb.shape[:2]
                    rect = Rectangle((0, 0), w - 1, h - 1, linewidth=4, edgecolor='green', facecolor='none', alpha=0.8)
                    ax2.add_patch(rect)
                    ax2.text(10, 30, 'GUIDE LANDMARK',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.8),
                             fontsize=12, fontweight='bold', color='white')
                else:
                    ax2.text(0.5, 0.5, 'Failed to load\nguide image', ha='center', va='center',
                             transform=ax2.transAxes)
            except Exception:
                ax2.text(0.5, 0.5, f'Error loading\nguide image', ha='center', va='center',
                         transform=ax2.transAxes)

            visible_landmark_id = inference_record['input']['visible_landmark_id']
            ax2.set_title(f'Visible Guide (L{visible_landmark_id})\n{Path(visible_reference_path).name}',
                          fontsize=12, fontweight='bold')
            ax2.axis('off')

            # Target landmark
            try:
                target_img = cv2.imread(target_reference_path)
                if target_img is not None:
                    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                    ax3.imshow(target_rgb)
                    h, w = target_rgb.shape[:2]
                    rect = Rectangle((0, 0), w - 1, h - 1, linewidth=4, edgecolor='red', facecolor='none', alpha=0.8)
                    ax3.add_patch(rect)
                    ax3.text(10, 30, 'TARGET LANDMARK',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                             fontsize=12, fontweight='bold', color='white')
                else:
                    ax3.text(0.5, 0.5, 'Failed to load\ntarget image', ha='center', va='center',
                             transform=ax3.transAxes)
            except Exception:
                ax3.text(0.5, 0.5, f'Error loading\ntarget image', ha='center', va='center',
                         transform=ax3.transAxes)

            target_landmark_id = inference_record['input']['target_landmark_id']
            ax3.set_title(f'Navigation Target (L{target_landmark_id})\n{Path(target_reference_path).name}',
                          fontsize=12, fontweight='bold')
            ax3.axis('off')

            # Prompt display
            ax4.axis('off')
            prompt_text = inference_record['input']['navigation_prompt']
            full_instruction = inference_record['input']['full_instruction']
            current_target = inference_record['input']['current_target']

            prompt_display = f"""DUAL-TARGET NAVIGATION INPUT

Navigation Instruction:
"{full_instruction}"

Current Target:
"{current_target}"

Dual-Target Strategy:
- Visible Guide: L{visible_landmark_id}
- True Target: L{target_landmark_id}

Model Configuration:
- Temperature: {self.temperature}
- Top-p: {self.top_p}
- Max tokens: {self.max_new_tokens}
- Sampling: {"Enabled" if self.do_sample else "Greedy"}
"""

            ax4.text(0.05, 0.95, prompt_display, transform=ax4.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
            ax4.set_title('Dual-Target Input & Config', fontsize=12, fontweight='bold')

            # Raw output
            ax5.axis('off')
            raw_output = inference_record['output']['raw_model_output']

            display_output = raw_output.replace('<|im_end|>', '').strip()
            if len(display_output) > 200:
                display_output = display_output[:200] + "..."

            raw_output_text = f"""RAW MODEL OUTPUT

{display_output}

Token Statistics:
- Input tokens: {inference_record['processing'].get('input_token_count', 'N/A')}
- Output tokens: {inference_record['processing'].get('output_token_count', 'N/A')}
- Generation time: {inference_record['timing'].get('generation_time', 0):.3f}s
"""

            ax5.text(0.05, 0.95, raw_output_text, transform=ax5.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            ax5.set_title('Raw Model Output', fontsize=12, fontweight='bold')

            # Processing steps
            ax6.axis('off')
            processing_text = f"""DUAL-TARGET PROCESSING

1. Image Loading:
   - Current view
   - Guide landmark  
   - Target landmark

2. Strategy Recognition:
   - Understand guide role
   - Identify true target
   - Plan exploration path

3. Decision Making:
   - Locate guide in view
   - Determine search direction
   - Balance approach & explore

Processing Time:
{inference_record['timing'].get('command_extraction_time', 0):.3f}s
"""

            ax6.text(0.05, 0.95, processing_text, transform=ax6.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
            ax6.set_title('Dual-Target Processing', fontsize=12, fontweight='bold')

            # Final command
            ax7.axis('off')
            final_command = inference_record['output']['final_command']
            action = final_command.get('action', 'UNKNOWN')
            reason = final_command.get('reason', 'No reason')
            confidence = final_command.get('confidence', 0.0)

            action_colors = {
                'MOVE_FORWARD': 'green',
                'TURN_LEFT': 'orange',
                'TURN_RIGHT': 'orange',
                'HOVER': 'red'
            }
            action_color = action_colors.get(action, 'gray')

            final_command_text = f"""DUAL-TARGET COMMAND

Action: {action}
Reason: {reason}
Confidence: {confidence:.2f}

Strategy:
Guide -> Target Navigation

Total Inference:
{inference_record['timing'].get('total_inference_time', 0):.3f}s

Performance Target:
< 0.333s (3Hz goal)

Status: {"REAL-TIME" if inference_record['timing'].get('total_inference_time', 1) < 0.333 else "SLOW"}
"""

            ax7.text(0.05, 0.95, final_command_text, transform=ax7.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=action_color, alpha=0.7))
            ax7.set_title('Navigation Command', fontsize=12, fontweight='bold')

            # Strategy analysis
            ax8.axis('off')
            strategy_text = f"""STRATEGY ANALYSIS

Dual-Target Benefits:
- Spatial reasoning active
- Context-aware navigation  
- Exploration vs tracking
- Semantic understanding

Navigation Logic:
1. Locate guide landmark
2. Understand spatial relation
3. Plan approach strategy
4. Search for true target

Landmark Relations:
- Guide: L{visible_landmark_id}
- Target: L{target_landmark_id}
- Path-based selection

Enhanced Capabilities:
- Beyond visual servoing
- Semantic path planning
- Multi-landmark reasoning
"""

            ax8.text(0.05, 0.95, strategy_text, transform=ax8.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            ax8.set_title('Strategy Analysis', fontsize=12, fontweight='bold')

            # Detailed statistics
            ax9.axis('off')

            timing = inference_record['timing']
            total_time = timing.get('total_inference_time', 0)

            time_components = [
                ('Image Processing', timing.get('image_processing_time', 0)),
                ('Tokenization', timing.get('tokenize_time', 0)),
                ('Model Generation', timing.get('generation_time', 0)),
                ('Decoding', timing.get('decode_time', 0)),
                ('Command Extraction', timing.get('command_extraction_time', 0)),
                ('Visualization', timing.get('visualization_time', 0))
            ]

            errors = inference_record.get('errors', [])
            error_text = "None" if not errors else "\n".join(f"- {err}" for err in errors)

            stats_text = f"""DUAL-TARGET NAVIGATION INFERENCE STATISTICS

Timing Breakdown (Total: {total_time:.3f}s):
"""

            timing_breakdown = []
            for name, time_val in time_components:
                percentage = (time_val / max(total_time, 0.001)) * 100
                timing_breakdown.append(f"   - {name}: {time_val:.3f}s ({percentage:.1f}%)")

            stats_text += "\n".join(timing_breakdown)

            stats_text += f"""

System Configuration:
   - Model: LLaVA-OneVision-Qwen2-7B
   - Strategy: DUAL-TARGET NAVIGATION
   - Quantization: {self.quantization_level if self.quantization_level > 0 else 'FP16'}
   - Flash Attention: Enabled
   - Device: {self.device}

Navigation Context:
   - Visible Guide: L{visible_landmark_id}
   - Navigation Target: L{target_landmark_id}
   - Landmark Sequence: {inference_record['input']['landmark_sequence']}

Errors: {error_text}

Files:
   - Current: {Path(current_image_path).name}
   - Guide: {Path(visible_reference_path).name}
   - Target: {Path(target_reference_path).name}
"""

            ax9.text(0.02, 0.98, stats_text, transform=ax9.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.6))

            plt.tight_layout()

            timestamp_safe = inference_record['timestamp'].replace(':', '-').replace('.', '_')
            viz_filename = f"dual_target_inference_{timestamp_safe}_{action}_{confidence:.2f}.png"
            viz_path = self.visualization_dir / viz_filename

            plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

        except Exception:
            pass

    def _save_inference_record(self, inference_record: dict, result: dict):
        """Save inference record for later analysis."""
        try:
            if not self.enable_visualization or not self.visualization_dir:
                return

            self.inference_history.append({
                'record': inference_record,
                'result': result
            })

            timestamp_safe = inference_record['timestamp'].replace(':', '-').replace('.', '_')
            record_filename = f"dual_target_record_{timestamp_safe}.json"
            record_path = self.visualization_dir / record_filename

            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'inference_record': inference_record,
                    'result': result
                }, f, indent=2, default=str, ensure_ascii=False)

            if len(self.inference_history) % 10 == 0:
                history_path = self.visualization_dir / "dual_target_inference_history.json"
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(self.inference_history, f, indent=2, default=str, ensure_ascii=False)

        except Exception:
            pass

    def get_inference_statistics(self) -> dict:
        """Get inference statistics."""
        if not self.inference_history:
            return {}

        total_inferences = len(self.inference_history)
        total_time = sum(record['record']['timing'].get('total_inference_time', 0)
                         for record in self.inference_history)
        avg_time = total_time / total_inferences if total_inferences > 0 else 0

        actions = [record['result'].get('action', 'UNKNOWN') for record in self.inference_history]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        confidences = [record['result'].get('confidence', 0) for record in self.inference_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        realtime_count = sum(1 for record in self.inference_history
                             if record['record']['timing'].get('total_inference_time', 1) < 0.333)
        realtime_rate = realtime_count / total_inferences if total_inferences > 0 else 0

        return {
            'total_inferences': total_inferences,
            'average_inference_time': avg_time,
            'realtime_performance_rate': realtime_rate,
            'action_distribution': action_counts,
            'average_confidence': avg_confidence,
            'total_processing_time': total_time,
            'navigation_strategy': 'dual_target'
        }

    def extract_navigation_command(self, full_output: str) -> dict:
        """Extract navigation command from model output with robust error handling."""
        clean_output = full_output.replace("<|im_end|>", "").strip()
        clean_lower = clean_output.lower()

        # Direct action keyword matching
        action_map = {
            'move_forward': 'MOVE_FORWARD',
            'forward': 'MOVE_FORWARD',
            'go': 'MOVE_FORWARD',
            'straight': 'MOVE_FORWARD',
            'ahead': 'MOVE_FORWARD',
            'turn_left': 'TURN_LEFT',
            'left': 'TURN_LEFT',
            'turn_right': 'TURN_RIGHT',
            'right': 'TURN_RIGHT',
            'hover': 'HOVER',
            'stop': 'HOVER',
            'wait': 'HOVER'
        }

        for keyword, action in action_map.items():
            if keyword in clean_lower:
                return {
                    "action": action,
                    "reason": f"Found '{keyword}'",
                    "confidence": 0.9
                }

        # Fallback: JSON parsing
        try:
            json_patterns = [
                r'\{"action":\s*"([^"]+)"',
                r'"action":\s*"([^"]+)"',
                r'action.*?([A-Z_]+)',
            ]

            for pattern in json_patterns:
                match = re.search(pattern, clean_output, re.IGNORECASE)
                if match:
                    action = match.group(1).upper()
                    valid_actions = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'HOVER']
                    if action in valid_actions:
                        return {
                            "action": action,
                            "reason": "JSON match",
                            "confidence": 0.8
                        }
        except:
            pass

        # Context-based inference
        if any(word in clean_lower for word in ['target', 'towards', 'approach']):
            return {
                "action": "MOVE_FORWARD",
                "reason": "Context suggests forward",
                "confidence": 0.6
            }

        # Default safe command
        return {
            "action": "HOVER",
            "reason": f"No clear action in: '{clean_output[:30]}'",
            "confidence": 0.0
        }