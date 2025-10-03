#!/usr/bin/env python3
"""
LLaVA-OneVision 3D Drone Navigation Evaluation Script.
Supports extended 3D action space and single GPU deployment.
"""

import os
import sys
import json
import torch
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.append('.')
sys.path.append('./LLaVA-NeXT')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, process_anyres_image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava import conversation as conversation_lib


class Drone3DNavigationEvaluator:
    """3D drone navigation task evaluator"""

    def __init__(self,
                 model_path: str,
                 base_model_path: str = None,
                 gpu_id: int = 0,
                 quantization_level: int = 0):
        """
        Initialize evaluator.

        Args:
            model_path: Fine-tuned model path (LoRA adapter path)
            base_model_path: Base model path, auto-inferred if None
            gpu_id: GPU ID to use
            quantization_level: Quantization level (0=FP16, 4=4bit, 8=8bit)
        """
        print("Initializing 3D Drone Navigation Evaluator...")

        self.model_path = Path(model_path)
        self.base_model_path = base_model_path
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.quantization_level = quantization_level

        self.action_mapping = {
            "START": 0,
            "MOVE_FORWARD": 1,
            "TURN_LEFT": 2,
            "TURN_RIGHT": 3,
            "MOVE_UP": 4,
            "MOVE_DOWN": 5,
            "STOP": 6
        }

        self.reverse_action_mapping = {v: k for k, v in self.action_mapping.items()}

        self.action_descriptions = {
            "START": "Start state",
            "MOVE_FORWARD": "Move forward 25cm",
            "TURN_LEFT": "Turn left 15 degrees",
            "TURN_RIGHT": "Turn right 15 degrees",
            "MOVE_UP": "Move up 20cm",
            "MOVE_DOWN": "Move down 20cm",
            "STOP": "Stop"
        }

        self.evaluation_results = []
        self.detailed_results = []

        self._load_model()
        print("Evaluator initialized successfully")

    def _load_model(self):
        """Load fine-tuned model with LoRA adapter support"""
        try:
            print(f"Loading model: {self.model_path}")
            print(f"Target device: {self.device}")

            is_lora_model = self._check_if_lora_model()

            if is_lora_model:
                print("LoRA model detected, loading with LoRA...")
                self._load_lora_model()
            else:
                print("Standard model detected, loading normally...")
                self._load_standard_model()

            if hasattr(self.model.config, "_attn_implementation") and self.model.config._attn_implementation == "flash_attention_2":
                print("Flash Attention 2 activated")

            self.model.eval()

            sample_param = next(self.model.parameters())
            self.model_dtype = sample_param.dtype
            print(f"Model dtype: {self.model_dtype}")

            self.vision_tower = self.model.get_vision_tower()
            if not self.vision_tower.is_loaded:
                self.vision_tower.load_model()
            self.vision_tower = self.vision_tower.to(device=self.device, dtype=self.model_dtype)

            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]

            print("Model loaded successfully")

        except Exception as e:
            print(f"Model loading failed: {e}")
            raise e

    def _check_if_lora_model(self) -> bool:
        """Check if model is LoRA format"""
        lora_files = [
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors"
        ]

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
        """Load LoRA model"""
        from peft import PeftModel

        if self.base_model_path is None:
            adapter_config_path = self.model_path / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                self.base_model_path = adapter_config.get("base_model_name_or_path")

        if self.base_model_path is None:
            self.base_model_path = "/root/autodl-tmp/model/transformers/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"

        print(f"Base model path: {self.base_model_path}")

        quantization_config = {}
        if self.quantization_level == 8:
            quantization_config['load_in_8bit'] = True
        elif self.quantization_level == 4:
            quantization_config['load_in_4bit'] = True

        self.tokenizer, base_model, self.image_processor, self.max_length = load_pretrained_model(
            model_path=self.base_model_path,
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

        print("LoRA model loaded")

    def _load_standard_model(self):
        """Load standard model"""
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

    def _prepare_conversation_template(self):
        """Prepare conversation template"""
        try:
            conv_template = conv_templates["qwen_1_5"].copy()
            conv_template.messages = []
            return conv_template
        except:
            print("Warning: Conversation template preparation failed")
            return None

    def _process_images_for_model(self, image_paths: List[str], data_base_dir: str) -> Tuple[List, List]:
        """
        Process image inputs for model inference.

        Args:
            image_paths: List of 3 image paths [current, visible, navigation]
            data_base_dir: Dataset base directory

        Returns:
            (image_tensors, image_sizes)
        """
        try:
            images = []
            image_sizes = []

            for img_path in image_paths:
                if not Path(img_path).is_absolute():
                    full_img_path = Path(data_base_dir) / img_path
                else:
                    full_img_path = Path(img_path)

                if not full_img_path.exists():
                    print(f"Warning: Image not found: {full_img_path}")
                    return None, None

                image = Image.open(full_img_path).convert('RGB')
                images.append(image)
                image_sizes.append(image.size)

            image_tensors = process_images(images, self.image_processor, self.model.config)

            if isinstance(image_tensors, list):
                image_tensors = [img.to(device=self.device, dtype=self.model_dtype) for img in image_tensors]
            else:
                image_tensors = image_tensors.to(device=self.device, dtype=self.model_dtype)

            return image_tensors, image_sizes

        except Exception as e:
            print(f"Image processing failed: {e}")
            return None, None

    def _create_evaluation_prompt(self, instruction: str = None) -> str:
        """
        Create evaluation prompt matching training format.
    
        Args:
            instruction: Navigation instruction (optional)
    
        Returns:
            Formatted prompt
        """
        if instruction:
            prompt = f"""Given the complete 3D navigation instruction and current situation, determine the next navigation action for the drone.
    
COMPLETE 3D NAVIGATION INSTRUCTION:
"{instruction}"
    
VISUAL INPUT:
Current drone observation: {DEFAULT_IMAGE_TOKEN}
Visible guide landmark: {DEFAULT_IMAGE_TOKEN} 
Target landmark for navigation: {DEFAULT_IMAGE_TOKEN}
    
Based on the complete instruction context and current 3D visual inputs, choose the most appropriate drone action from the available 3D action space:"""
        else:
            prompt = f"""Given the current 3D navigation situation, determine the next navigation action for the drone.
    
VISUAL INPUT:
Current drone observation: {DEFAULT_IMAGE_TOKEN}
Visible guide landmark: {DEFAULT_IMAGE_TOKEN} 
Target landmark for navigation: {DEFAULT_IMAGE_TOKEN}
    
Based on the current 3D visual inputs, choose the most appropriate drone action from the available 3D action space:"""
    
        return prompt

    def _extract_instruction_from_prompt(self, human_input: str) -> str:
        """Extract navigation instruction from human input"""
        try:
            instruction_start = human_input.find('COMPLETE 3D NAVIGATION INSTRUCTION:\n"')
            if instruction_start != -1:
                instruction_start += len('COMPLETE 3D NAVIGATION INSTRUCTION:\n"')
                instruction_end = human_input.find('"', instruction_start)
                if instruction_end != -1:
                    return human_input[instruction_start:instruction_end].strip()
            
            return ""
        except Exception:
            return ""

    def _extract_action_from_output(self, raw_output: str) -> Tuple[str, str]:
        """
        Extract action from model output.

        Args:
            raw_output: Raw model output

        Returns:
            (predicted_action, cleaned_output)
        """
        try:
            cleaned_output = raw_output.replace("<|im_end|>", "").strip()

            for action_name in self.action_mapping.keys():
                if action_name in cleaned_output.upper():
                    return action_name, cleaned_output

            cleaned_lower = cleaned_output.lower()
            if any(word in cleaned_lower for word in ['stop', 'wait', 'halt']):
                return "STOP", cleaned_output
            elif any(word in cleaned_lower for word in ['forward', 'ahead', 'straight', 'move']):
                return "MOVE_FORWARD", cleaned_output
            elif any(word in cleaned_lower for word in ['left', 'turn_left']):
                return "TURN_LEFT", cleaned_output
            elif any(word in cleaned_lower for word in ['right', 'turn_right']):
                return "TURN_RIGHT", cleaned_output
            elif any(word in cleaned_lower for word in ['up', 'ascend', 'rise']):
                return "MOVE_UP", cleaned_output
            elif any(word in cleaned_lower for word in ['down', 'descend', 'lower']):
                return "MOVE_DOWN", cleaned_output
            elif any(word in cleaned_lower for word in ['start', 'begin']):
                return "START", cleaned_output

            return "STOP", cleaned_output

        except Exception as e:
            print(f"Warning: Action extraction failed: {e}")
            return "STOP", str(raw_output)

    def evaluate_single_sample(self, sample: Dict, data_base_dir: str) -> Dict:
        """
        Evaluate single sample.

        Args:
            sample: Sample containing id, image, conversations
            data_base_dir: Dataset base directory

        Returns:
            Evaluation result dictionary
        """
        try:
            sample_id = sample.get('id', 'unknown')
            images = sample.get('image', [])
            conversations = sample.get('conversations', [])

            if len(images) != 3:
                return {
                    'sample_id': sample_id,
                    'success': False,
                    'error': f'Expected 3 images, got {len(images)}',
                    'predicted_action': 'ERROR',
                    'true_action': 'UNKNOWN',
                    'correct': False,
                    'raw_output': '',
                    'confidence': 0.0
                }

            if len(conversations) < 2:
                return {
                    'sample_id': sample_id,
                    'success': False,
                    'error': 'Invalid conversations format',
                    'predicted_action': 'ERROR',
                    'true_action': 'UNKNOWN',
                    'correct': False,
                    'raw_output': '',
                    'confidence': 0.0
                }

            human_input = conversations[0]['value']
            true_action = conversations[1]['value']

            scene_id, current_frame_id, navigation_frame_id = self._extract_scene_info_from_prompt(human_input)
            instruction = self._extract_instruction_from_prompt(human_input)

            image_tensors, image_sizes = self._process_images_for_model(images, data_base_dir)
            if image_tensors is None:
                return {
                    'sample_id': sample_id,
                    'success': False,
                    'error': 'Image processing failed',
                    'predicted_action': 'ERROR',
                    'true_action': true_action,
                    'correct': False,
                    'raw_output': '',
                    'confidence': 0.0
                }

            eval_prompt = self._create_evaluation_prompt(instruction)

            conv = self._prepare_conversation_template()
            if conv is None:
                return {
                    'sample_id': sample_id,
                    'success': False,
                    'error': 'Conversation template failed',
                    'predicted_action': 'ERROR',
                    'true_action': true_action,
                    'correct': False,
                    'raw_output': '',
                    'confidence': 0.0
                }

            conv.append_message(conv.roles[0], eval_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.device)

            start_time = time.time()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=32,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            inference_time = time.time() - start_time

            raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            predicted_action, cleaned_output = self._extract_action_from_output(raw_output)

            correct = (predicted_action == true_action)

            confidence = self._calculate_confidence(raw_output, predicted_action)

            return {
                'sample_id': sample_id,
                'success': True,
                'instruction': instruction,
                'current_frame_id': current_frame_id,
                'navigation_frame_id': navigation_frame_id,
                'predicted_action': predicted_action,
                'true_action': true_action,
                'correct': correct,
                'raw_output': raw_output,
                'cleaned_output': cleaned_output,
                'confidence': confidence,
                'inference_time': inference_time,
                'error': None
            }

        except Exception as e:
            return {
                'sample_id': sample.get('id', 'unknown'),
                'success': False,
                'error': str(e),
                'predicted_action': 'ERROR',
                'true_action': sample.get('conversations', [{}])[1].get('value', 'UNKNOWN') if len(
                    sample.get('conversations', [])) > 1 else 'UNKNOWN',
                'correct': False,
                'raw_output': '',
                'confidence': 0.0
            }

    def _extract_scene_info_from_prompt(self, human_input: str) -> Tuple[str, str, str]:
        """Extract scene information from human input"""
        try:
            scene_id = "unknown_scene"
            current_frame_id = "unknown_frame"
            navigation_frame_id = "unknown_target"

            scene_start = human_input.find('Scene: ')
            if scene_start != -1:
                scene_start += len('Scene: ')
                scene_end = human_input.find('\n', scene_start)
                if scene_end != -1:
                    scene_id = human_input[scene_start:scene_end].strip()

            current_start = human_input.find('Current position: Frame ')
            if current_start != -1:
                current_start += len('Current position: Frame ')
                current_end = human_input.find('\n', current_start)
                if current_end != -1:
                    current_frame_id = human_input[current_start:current_end].strip()

            nav_start = human_input.find('Navigation target: Frame ')
            if nav_start != -1:
                nav_start += len('Navigation target: Frame ')
                nav_end = human_input.find('\n', nav_start)
                if nav_end != -1:
                    navigation_frame_id = human_input[nav_start:nav_end].strip()

            return scene_id, current_frame_id, navigation_frame_id

        except Exception:
            return "unknown_scene", "unknown_frame", "unknown_target"

    def _calculate_confidence(self, raw_output: str, predicted_action: str) -> float:
        """Calculate prediction confidence"""
        try:
            if predicted_action in raw_output.upper():
                return 0.9
            elif predicted_action != "STOP":
                return 0.7
            else:
                return 0.5
        except Exception:
            return 0.0

    def evaluate_dataset(self,
                         data_path: str,
                         max_samples: Optional[int] = None,
                         output_dir: str = "./evaluation_results") -> Dict:
        """
        Evaluate entire dataset.

        Args:
            data_path: Test data JSON file path
            max_samples: Maximum samples to evaluate (None for all)
            output_dir: Results output directory

        Returns:
            Evaluation statistics
        """
        print(f"Starting dataset evaluation: {data_path}")

        data_file_path = Path(data_path)
        data_base_dir = data_file_path.parent
        print(f"Data base directory: {data_base_dir}")

        images_dir = data_base_dir / "images"
        if not images_dir.exists():
            print(f"Warning: Images directory not found: {images_dir}")
        else:
            print(f"Images directory found: {images_dir}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            print(f"Loaded {len(test_data)} test samples")
        except Exception as e:
            print(f"Data loading failed: {e}")
            return {}

        if max_samples is not None:
            test_data = test_data[:max_samples]
            print(f"Limited to {max_samples} samples")

        total_samples = len(test_data)
        correct_predictions = 0
        action_correct_counts = {action: 0 for action in self.action_mapping.keys()}
        action_total_counts = {action: 0 for action in self.action_mapping.keys()}
        confusion_matrix = {}

        detailed_results = []
        start_time = time.time()

        for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
            result = self.evaluate_single_sample(sample, str(data_base_dir))
            detailed_results.append(result)

            if result['success'] and result['correct']:
                correct_predictions += 1

            true_action = result['true_action']
            pred_action = result['predicted_action']

            if true_action in action_total_counts:
                action_total_counts[true_action] += 1

                if result['correct']:
                    action_correct_counts[true_action] += 1

            if true_action not in confusion_matrix:
                confusion_matrix[true_action] = {}
            if pred_action not in confusion_matrix[true_action]:
                confusion_matrix[true_action][pred_action] = 0
            confusion_matrix[true_action][pred_action] += 1

            if (i + 1) % 10 == 0:
                current_accuracy = correct_predictions / (i + 1)
                success_rate = sum(1 for r in detailed_results if r['success']) / (i + 1)
                print(f"Progress {i + 1}/{total_samples}, Accuracy: {current_accuracy:.3f}, Success: {success_rate:.3f}")

        total_time = time.time() - start_time

        successful_samples = [r for r in detailed_results if r['success']]
        overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        success_rate = len(successful_samples) / total_samples if total_samples > 0 else 0

        if successful_samples:
            avg_inference_time = np.mean([r['inference_time'] for r in successful_samples])
        else:
            avg_inference_time = 0.0

        action_accuracies = {}
        for action in self.action_mapping.keys():
            if action_total_counts[action] > 0:
                action_accuracies[action] = action_correct_counts[action] / action_total_counts[action]
            else:
                action_accuracies[action] = 0.0

        evaluation_stats = {
            'overall_metrics': {
                'total_samples': total_samples,
                'successful_samples': len(successful_samples),
                'success_rate': success_rate,
                'correct_predictions': correct_predictions,
                'overall_accuracy': overall_accuracy,
                'avg_inference_time': avg_inference_time,
                'total_evaluation_time': total_time
            },
            'action_metrics': {
                'action_accuracies': action_accuracies,
                'action_counts': action_total_counts,
                'action_correct_counts': action_correct_counts,
                'action_descriptions': self.action_descriptions
            },
            'confusion_matrix': confusion_matrix,
            'model_info': {
                'model_path': str(self.model_path),
                'device': self.device,
                'quantization_level': self.quantization_level,
                'model_dtype': str(self.model_dtype),
                'action_space_size': len(self.action_mapping)
            },
            'evaluation_config': {
                'data_path': data_path,
                'data_base_dir': str(data_base_dir),
                'max_samples': max_samples,
                'timestamp': datetime.now().isoformat()
            }
        }

        detailed_results_path = output_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        stats_path = output_dir / f"evaluation_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_stats, f, indent=2, ensure_ascii=False)

        self._print_evaluation_results(evaluation_stats)

        print(f"Detailed results saved: {detailed_results_path}")
        print(f"Statistics saved: {stats_path}")

        return evaluation_stats

    def _print_evaluation_results(self, stats: Dict):
        """Print evaluation results"""
        print("\n" + "=" * 80)
        print("3D Drone Navigation Evaluation Results")
        print("=" * 80)

        overall = stats['overall_metrics']
        action_metrics = stats['action_metrics']

        print(f"\nOverall Performance:")
        print(f"  Total samples: {overall['total_samples']:,}")
        print(f"  Successful samples: {overall['successful_samples']:,}")
        print(f"  Success rate: {overall['success_rate']:.4f} ({overall['success_rate'] * 100:.2f}%)")
        print(f"  Correct predictions: {overall['correct_predictions']:,}")
        print(f"  Overall accuracy: {overall['overall_accuracy']:.4f} ({overall['overall_accuracy'] * 100:.2f}%)")
        print(f"  Avg inference time: {overall['avg_inference_time']:.4f}s")
        print(f"  Total evaluation time: {overall['total_evaluation_time']:.2f}s")

        print(f"\n3D Action Space Accuracy:")
        action_descriptions = action_metrics['action_descriptions']
        for action, accuracy in action_metrics['action_accuracies'].items():
            count = action_metrics['action_counts'][action]
            correct = action_metrics['action_correct_counts'][action]
            description = action_descriptions.get(action, "Unknown")
            action_type = "3D Extended" if action in ["MOVE_UP", "MOVE_DOWN"] else "Original 2D"
            print(f"  {action:12}: {accuracy:.4f} ({accuracy * 100:5.2f}%) [{correct:4}/{count:4}] - {description} ({action_type})")

        print(f"\nConfusion Matrix:")
        confusion = stats['confusion_matrix']
        actions = sorted(self.action_mapping.keys())

        header_text = "True\\Pred"
        print(f"{header_text:>12}", end="")
        for action in actions:
            print(f"{action:>12}", end="")
        print()

        for true_action in actions:
            print(f"{true_action:>12}", end="")
            for pred_action in actions:
                count = confusion.get(true_action, {}).get(pred_action, 0)
                print(f"{count:>12}", end="")
            print()

        drone_3d_actions = ["MOVE_UP", "MOVE_DOWN"]
        total_3d_samples = sum(action_metrics['action_counts'].get(action, 0) for action in drone_3d_actions)
        correct_3d_samples = sum(action_metrics['action_correct_counts'].get(action, 0) for action in drone_3d_actions)

        if total_3d_samples > 0:
            drone_3d_accuracy = correct_3d_samples / total_3d_samples
            print(f"\n3D-Specific Action Performance:")
            print(f"  Total 3D action samples: {total_3d_samples:,}")
            print(f"  Correct 3D predictions: {correct_3d_samples:,}")
            print(f"  3D action accuracy: {drone_3d_accuracy:.4f} ({drone_3d_accuracy * 100:.2f}%)")

        print("\n" + "=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LLaVA-OneVision 3D Drone Navigation Model Evaluation')

    parser.add_argument('--model_path',
                        default='./drone_3d_navigation_finetune_output_4gpu_fix',
                        help='Fine-tuned model path (LoRA adapter path)')
    parser.add_argument('--base_model_path',
                        default=None,
                        help='Base model path (auto-inferred if not specified)')
    parser.add_argument('--test_data',
                        default='./llava_drone_3d_navigation_data_ins/drone_3d_navigation_val_subset_1000.json',
                        help='Test data JSON file path')
    parser.add_argument('--output_dir',
                        default='./evaluation_results_fix',
                        help='Evaluation results output directory')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='GPU ID to use')
    parser.add_argument('--max_samples',
                        type=int,
                        default=None,
                        help='Maximum samples to evaluate (None for all)')
    parser.add_argument('--quantization',
                        type=int,
                        choices=[0, 4, 8],
                        default=0,
                        help='Quantization level (0=FP16, 4=4bit, 8=8bit)')

    args = parser.parse_args()

    print("Starting 3D Drone Navigation Evaluation")
    print(f"Model path: {args.model_path}")
    if args.base_model_path:
        print(f"Base model path: {args.base_model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Quantization level: {args.quantization}")

    if not Path(args.model_path).exists():
        print(f"Error: Model path not found: {args.model_path}")
        return

    if not Path(args.test_data).exists():
        print(f"Error: Test data not found: {args.test_data}")
        return

    try:
        evaluator = Drone3DNavigationEvaluator(
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            gpu_id=args.gpu_id,
            quantization_level=args.quantization
        )

        evaluation_stats = evaluator.evaluate_dataset(
            data_path=args.test_data,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
