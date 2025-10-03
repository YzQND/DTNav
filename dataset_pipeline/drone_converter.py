#!/usr/bin/env python3
"""
Convert 3D drone navigation data to LLaVA format.
Supports 3D action space with vertical movements (move_up, move_down).
Creates balanced datasets with configurable subset sizes.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
import shutil
import random
from collections import defaultdict


class Drone3DNavigationDataConverter:
    """Converter for 3D drone navigation data to LLaVA format"""

    def __init__(self, drone_targets_json: str, drone_base_path: str, annotations_json: str, output_dir: str = "./llava_drone_3d_navigation_data"):
        self.drone_targets_json = drone_targets_json
        self.drone_base_path = Path(drone_base_path)
        self.annotations_json = annotations_json
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.action_mapping = {
            None: "START",
            "start": "START",
            "move_forward": "MOVE_FORWARD",
            "turn_left": "TURN_LEFT", 
            "turn_right": "TURN_RIGHT",
            "move_up": "MOVE_UP",
            "move_down": "MOVE_DOWN",
            "stop": "STOP"
        }

        self.balanced_actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_UP", "MOVE_DOWN", "STOP"]

        self.trajectory_instructions = self._load_trajectory_instructions()
        self.placeholder_image_path = self._create_placeholder_image()

        print(f"Converter initialized: {len(self.trajectory_instructions)} trajectory instructions loaded")

    def _load_trajectory_instructions(self) -> dict:
        """Load trajectory instructions from annotations.json"""
        trajectory_instructions = {}
        
        try:
            with open(self.annotations_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'trajectories' in data:
                trajectories = data['trajectories']
            elif isinstance(data, list):
                trajectories = data
            else:
                trajectories = [data] if isinstance(data, dict) else []
            
            for trajectory in trajectories:
                path_id = trajectory.get('path_id')
                instruction = trajectory.get('instruction', '')
                
                if path_id and instruction:
                    trajectory_instructions[path_id] = instruction
                    
        except Exception as e:
            print(f"Warning: Failed to load trajectory instructions: {e}")
            
        return trajectory_instructions

    def _get_instruction_for_trajectory(self, trajectory_id: str) -> str:
        """Get navigation instruction for specified trajectory"""
        instruction = self.trajectory_instructions.get(trajectory_id, "")
        
        if not instruction:
            instruction = f"Navigate through the 3D environment to reach the target destination in trajectory {trajectory_id}."
                
        return instruction

    def _create_placeholder_image(self) -> str:
        """Create 384x384 gray placeholder image"""
        try:
            placeholder_img = Image.new('RGB', (384, 384), (128, 128, 128))
            placeholder_path = self.images_dir / "placeholder_visible_landmark.jpg"
            placeholder_img.save(placeholder_path, 'JPEG')
            return f"images/placeholder_visible_landmark.jpg"
        except Exception as e:
            print(f"Error creating placeholder image: {e}")
            return None

    def _copy_drone_image(self, path_id: str, frame_id: str, image_type: str) -> str:
        """Copy drone image from dataset to output directory"""
        try:
            if '_' in frame_id:
                frame_idx = frame_id.split('_')[-1]
            else:
                frame_idx = frame_id.replace('.jpg', '').replace('.png', '')
            
            try:
                frame_num = int(frame_idx)
                source_image_filename = f"{frame_num:04d}_rgb.png"
            except ValueError:
                source_image_filename = frame_id if frame_id.endswith(('.jpg', '.png')) else f"{frame_id}.png"
            
            source_image_path = self.drone_base_path / path_id / source_image_filename

            if not source_image_path.exists():
                return None

            trajectory_id = path_id.replace('/', '_').replace('\\', '_')
            filename = f"{trajectory_id}_{frame_id.replace('.jpg', '').replace('.png', '')}_{image_type}.jpg"
            target_path = self.images_dir / filename

            if source_image_path.suffix.lower() == '.png':
                img = Image.open(source_image_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(target_path, 'JPEG', quality=95)
            else:
                shutil.copy2(source_image_path, target_path)

            return f"images/{filename}"

        except Exception as e:
            return None

    def _copy_landmark_image(self, landmark_path: str, trajectory_id: str, frame_id: str, landmark_type: str) -> str:
        """Copy landmark image to output directory"""
        try:
            if not landmark_path or not Path(landmark_path).exists():
                return None

            ext = Path(landmark_path).suffix
            safe_trajectory_id = trajectory_id.replace('/', '_').replace('\\', '_')
            filename = f"{safe_trajectory_id}_{frame_id.replace('.jpg', '').replace('.png', '')}_{landmark_type}{ext}"
            new_path = self.images_dir / filename

            shutil.copy2(landmark_path, new_path)
            return f"images/{filename}"

        except Exception as e:
            return None

    def _create_subset_data(self, train_data: list, val_data: list, subset_train_size: int, subset_val_size: int):
        """Create random subset from full dataset"""
        actual_train_size = min(subset_train_size, len(train_data))
        actual_val_size = min(subset_val_size, len(val_data))

        random.shuffle(train_data)
        random.shuffle(val_data)

        subset_train_data = train_data[:actual_train_size]
        subset_val_data = val_data[:actual_val_size]

        subset_train_path = self.output_dir / f"drone_3d_navigation_train_subset_{actual_train_size}.json"
        subset_val_path = self.output_dir / f"drone_3d_navigation_val_subset_{actual_val_size}.json"

        with open(subset_train_path, 'w', encoding='utf-8') as f:
            json.dump(subset_train_data, f, indent=2, ensure_ascii=False)

        with open(subset_val_path, 'w', encoding='utf-8') as f:
            json.dump(subset_val_data, f, indent=2, ensure_ascii=False)

        print(f"Created random subset: train={actual_train_size}, val={actual_val_size}")

        self._generate_subset_statistics(subset_train_data, subset_val_data, actual_train_size, actual_val_size, "random")

        return str(subset_train_path), str(subset_val_path)

    def _create_balanced_subset_data(self, train_data: list, val_data: list, subset_train_size: int, subset_val_size: int):
        """Create balanced subset: prioritize STOP actions, distribute remaining capacity evenly"""
        all_data = train_data + val_data

        action_groups = defaultdict(list)
        for item in all_data:
            action = item["conversations"][1]["value"]
            if action in self.balanced_actions:
                action_groups[action].append(item)

        # Allocate STOP data: 90% train, 10% val
        stop_data = action_groups.get("STOP", [])
        random.shuffle(stop_data)

        stop_train_count = int(len(stop_data) * 0.9)
        stop_val_count = len(stop_data) - stop_train_count

        stop_train_data = stop_data[:stop_train_count]
        stop_val_data = stop_data[stop_train_count:]

        train_action_groups = defaultdict(list)
        val_action_groups = defaultdict(list)

        train_action_groups["STOP"] = stop_train_data
        val_action_groups["STOP"] = stop_val_data

        for action in ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_UP", "MOVE_DOWN"]:
            original_data = action_groups.get(action, [])
            train_action_groups[action] = original_data.copy()
            val_action_groups[action] = original_data.copy()

        balanced_train_data = self._create_single_balanced_subset(train_action_groups, subset_train_size, "train")

        used_non_stop_ids = {item["id"] for item in balanced_train_data if item["conversations"][1]["value"] != "STOP"}

        remaining_val_action_groups = defaultdict(list)
        remaining_val_action_groups["STOP"] = val_action_groups["STOP"]

        for action in ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_UP", "MOVE_DOWN"]:
            remaining_val_action_groups[action] = [
                item for item in val_action_groups[action]
                if item["id"] not in used_non_stop_ids
            ]

        balanced_val_data = self._create_single_balanced_subset(remaining_val_action_groups, subset_val_size, "val")

        balanced_train_path = self.output_dir / f"drone_3d_navigation_train_balanced_{subset_train_size}.json"
        balanced_val_path = self.output_dir / f"drone_3d_navigation_val_balanced_{subset_val_size}.json"

        with open(balanced_train_path, 'w', encoding='utf-8') as f:
            json.dump(balanced_train_data, f, indent=2, ensure_ascii=False)

        with open(balanced_val_path, 'w', encoding='utf-8') as f:
            json.dump(balanced_val_data, f, indent=2, ensure_ascii=False)

        print(f"Created balanced subset: train={subset_train_size}, val={subset_val_size}")

        self._generate_subset_statistics(balanced_train_data, balanced_val_data, subset_train_size, subset_val_size, "balanced")

        return str(balanced_train_path), str(balanced_val_path)

    def _create_single_balanced_subset(self, action_groups: dict, target_size: int, dataset_name: str) -> list:
        """Create single balanced subset"""
        stop_data = action_groups.get("STOP", [])
        stop_count = len(stop_data)

        if stop_count >= target_size:
            random.shuffle(stop_data)
            return stop_data[:target_size]

        remaining_capacity = target_size - stop_count
        other_actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_UP", "MOVE_DOWN"]
        per_action_count = remaining_capacity // len(other_actions)
        extra_count = remaining_capacity % len(other_actions)

        balanced_data = stop_data.copy()

        for i, action in enumerate(other_actions):
            available_data = action_groups.get(action, [])
            needed_count = per_action_count + (1 if i < extra_count else 0)
            actual_count = min(needed_count, len(available_data))

            if actual_count > 0:
                random.shuffle(available_data)
                balanced_data.extend(available_data[:actual_count])

        random.shuffle(balanced_data)
        return balanced_data

    def _generate_subset_statistics(self, subset_train_data: list, subset_val_data: list, train_size: int, val_size: int, subset_type: str):
        """Generate subset statistics report"""
        total_subset_samples = len(subset_train_data) + len(subset_val_data)

        subset_action_counts = {action: 0 for action in self.action_mapping.values()}
        for item in subset_train_data + subset_val_data:
            action = item["conversations"][1]["value"]
            if action in subset_action_counts:
                subset_action_counts[action] += 1

        subset_stats = {
            "subset_info": {
                "subset_name": f"Drone3D_subset_{subset_type}_{train_size}+{val_size}",
                "subset_type": subset_type,
                "subset_train_samples": len(subset_train_data),
                "subset_val_samples": len(subset_val_data),
                "subset_total_samples": total_subset_samples,
                "subset_creation_method": f"{subset_type}_sampling"
            },
            "subset_action_distribution": subset_action_counts
        }

        subset_stats_path = self.output_dir / f"drone_3d_subset_{subset_type}_{train_size}+{val_size}_statistics.json"
        with open(subset_stats_path, 'w', encoding='utf-8') as f:
            json.dump(subset_stats, f, indent=2, ensure_ascii=False)

    def convert_to_llava_format(self, subset_train_size: int = None, subset_val_size: int = None, create_balanced: bool = False):
        """Convert 3D drone data to LLaVA format with optional subset creation"""
        print("Starting conversion to LLaVA format...")

        with open(self.drone_targets_json, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        llava_data = []

        total_count = 0
        missing_basic_info_count = 0
        missing_current_image_count = 0
        missing_navigation_target_count = 0
        missing_visible_target_count = 0
        missing_instruction_count = 0
        successful_count = 0

        target_pairs = json_data.get('target_pairs', [])

        for pair in tqdm(target_pairs, desc="Converting frames"):
            total_count += 1

            try:
                trajectory_id = pair.get('trajectory_id', '')
                scene_id = pair.get('scene_id', '')
                current_frame_id = pair.get('current_frame_id', '')
                navigation_frame_id = pair.get('navigation_frame_id', '')
                current_action = pair.get('current_action', None)

                if not all([trajectory_id, scene_id, current_frame_id, navigation_frame_id]):
                    missing_basic_info_count += 1
                    continue

                instruction = self._get_instruction_for_trajectory(trajectory_id)
                if not instruction:
                    missing_instruction_count += 1
                    continue

                current_image_path = self._copy_drone_image(trajectory_id, current_frame_id, 'current')

                if not current_image_path:
                    missing_current_image_count += 1
                    continue

                navigation_target_path = None
                if pair.get('navigation_target_path') and pair.get('navigation_target_id'):
                    navigation_target_path = self._copy_landmark_image(
                        pair['navigation_target_path'],
                        trajectory_id,
                        current_frame_id,
                        'navigation_landmark'
                    )

                if not navigation_target_path:
                    missing_navigation_target_count += 1
                    continue

                visible_landmark_path = None

                if (pair.get('visible_target_path') and pair.get('visible_target_id') and
                        Path(pair['visible_target_path']).exists()):
                    visible_landmark_path = self._copy_landmark_image(
                        pair['visible_target_path'],
                        trajectory_id,
                        current_frame_id,
                        'visible_landmark'
                    )

                if not visible_landmark_path:
                    visible_landmark_path = self.placeholder_image_path
                    missing_visible_target_count += 1

                images = [current_image_path, visible_landmark_path, navigation_target_path]

                prompt = f"""Given the complete 3D navigation instruction and current situation, determine the next navigation action for the drone.

COMPLETE 3D NAVIGATION INSTRUCTION:
"{instruction}"

CURRENT NAVIGATION CONTEXT:
Current observation: <image>
Visible guide landmark: <image> 
Current subgoal target: <image>

Based on the complete instruction context and current visual inputs, choose the most appropriate 3D action space:"""

                action_label = self.action_mapping.get(current_action, "UNKNOWN_ACTION")

                conversation = [
                    {
                        "from": "human",
                        "value": prompt
                    },
                    {
                        "from": "gpt",
                        "value": action_label
                    }
                ]

                llava_item = {
                    "id": f"{trajectory_id}_{current_frame_id.replace('.jpg', '').replace('.png', '')}",
                    "image": images,
                    "conversations": conversation
                }

                llava_data.append(llava_item)
                successful_count += 1

            except Exception as e:
                continue

        print(f"\nConversion summary:")
        print(f"  Total: {total_count}, Successful: {successful_count}")
        print(f"  Missing: basic_info={missing_basic_info_count}, instruction={missing_instruction_count}")
        print(f"  Missing: current_image={missing_current_image_count}, nav_target={missing_navigation_target_count}")
        print(f"  Missing visible_target (using placeholder): {missing_visible_target_count}")

        random.shuffle(llava_data)
        split_idx = int(len(llava_data) * 0.9)

        train_data = llava_data[:split_idx]
        val_data = llava_data[split_idx:]

        train_path = self.output_dir / "drone_3d_navigation_train_full.json"
        val_path = self.output_dir / "drone_3d_navigation_val_full.json"

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        print(f"\nFull dataset saved:")
        print(f"  Train: {train_path} ({len(train_data)} samples)")
        print(f"  Val: {val_path} ({len(val_data)} samples)")

        subset_train_path = None
        subset_val_path = None
        balanced_train_path = None
        balanced_val_path = None

        if subset_train_size is not None and subset_val_size is not None:
            if subset_train_size > 0 and subset_val_size > 0:
                subset_train_path, subset_val_path = self._create_subset_data(
                    train_data.copy(), val_data.copy(), subset_train_size, subset_val_size
                )

                if create_balanced:
                    balanced_train_path, balanced_val_path = self._create_balanced_subset_data(
                        train_data.copy(), val_data.copy(), subset_train_size, subset_val_size
                    )

        original_train_path = self.output_dir / "drone_3d_navigation_train.json"
        original_val_path = self.output_dir / "drone_3d_navigation_val.json"

        if subset_train_path and subset_val_path:
            shutil.copy2(subset_train_path, original_train_path)
            shutil.copy2(subset_val_path, original_val_path)
            print(f"\nDefault files point to random subset")
        else:
            shutil.copy2(train_path, original_train_path)
            shutil.copy2(val_path, original_val_path)
            print(f"\nDefault files point to full dataset")

        self._generate_statistics(
            train_data, val_data, total_count, missing_basic_info_count,
            missing_current_image_count, missing_navigation_target_count,
            missing_visible_target_count, missing_instruction_count, successful_count
        )

        return str(original_train_path), str(original_val_path)

    def _generate_statistics(self, train_data, val_data, total_count, missing_basic_info_count,
                             missing_current_image_count, missing_navigation_target_count,
                             missing_visible_target_count, missing_instruction_count, successful_count):
        """Generate dataset statistics report"""
        total_samples = len(train_data) + len(val_data)

        stats = {
            "dataset_info": {
                "dataset_name": "Drone3D",
                "total_input_samples": total_count,
                "successful_samples": successful_count,
                "full_train_samples": len(train_data),
                "full_val_samples": len(val_data),
                "format": "LLaVA conversation format with 3 images for 3D navigation",
                "action_space": "3D Extended (7 actions)",
                "supported_actions": list(self.action_mapping.values()),
                "success_rate": f"{(successful_count / total_count * 100):.1f}%" if total_count > 0 else "0%",
                "instruction_source": "Original annotations.json"
            },
            "filtering_statistics": {
                "missing_basic_info": missing_basic_info_count,
                "missing_instruction": missing_instruction_count,
                "missing_current_image": missing_current_image_count,
                "missing_navigation_target": missing_navigation_target_count,
                "missing_visible_target": missing_visible_target_count,
                "placeholder_usage_rate": f"{(missing_visible_target_count / successful_count * 100):.1f}%" if successful_count > 0 else "0%"
            },
            "data_distribution": {},
            "image_composition": {
                "images_per_sample": 3,
                "image_types": ["current_drone_observation", "visible_landmark", "navigation_target"],
                "placeholder_used": missing_visible_target_count
            },
            "action_space_extension": {
                "original_2d_actions": ["START", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"],
                "added_3d_actions": ["MOVE_UP", "MOVE_DOWN"],
                "total_action_space": len(self.action_mapping)
            }
        }

        action_counts = {action: 0 for action in self.action_mapping.values()}

        for item in train_data + val_data:
            action = item["conversations"][1]["value"]
            if action in action_counts:
                action_counts[action] += 1

        stats["data_distribution"] = action_counts

        stats_path = self.output_dir / "drone_3d_conversion_statistics_full.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"\nDataset statistics:")
        print(f"  Success rate: {stats['dataset_info']['success_rate']}")
        print(f"  Placeholder usage: {stats['filtering_statistics']['placeholder_usage_rate']}")
        print(f"  Images per sample: {stats['image_composition']['images_per_sample']}")
        print(f"  Action space size: {stats['action_space_extension']['total_action_space']}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Convert 3D drone navigation data to LLaVA format')
    parser.add_argument('--drone_targets',
                        default='results/landmark_navigation_20250914_201303/drone_3d_training_targets_optimized.json',
                        help='Drone target pairs JSON file path')
    parser.add_argument('--drone_base_path',
                        default='./drone_navigation_data',
                        help='Drone dataset base path')
    parser.add_argument('--annotations_json',
                        default='./drone_navigation_data/annotations.json',
                        help='Original annotations file path')
    parser.add_argument('--output_dir',
                        default='./llava_drone_3d_navigation_data_ins',
                        help='Output directory')
    parser.add_argument('--subset_train_size',
                        type=int,
                        default=None,
                        help='Training subset size (optional)')
    parser.add_argument('--subset_val_size',
                        type=int,
                        default=None,
                        help='Validation subset size (optional)')
    parser.add_argument('--create_balanced',
                        action='store_true',
                        help='Create balanced dataset (prioritize STOP actions, distribute others evenly)')

    args = parser.parse_args()

    print("=" * 70)
    print("Convert 3D Drone Navigation Data to LLaVA Format")
    print("=" * 70)
    print(f"Input file: {args.drone_targets}")
    print(f"Drone data path: {args.drone_base_path}")
    print(f"Annotations file: {args.annotations_json}")
    print(f"Output directory: {args.output_dir}")

    if args.subset_train_size is not None and args.subset_val_size is not None:
        print(f"\nSubset configuration:")
        print(f"  Train subset size: {args.subset_train_size}")
        print(f"  Val subset size: {args.subset_val_size}")
        print(f"  Create balanced: {args.create_balanced}")

    if not Path(args.drone_targets).exists():
        print(f"Error: Input file not found: {args.drone_targets}")
        return

    if not Path(args.drone_base_path).exists():
        print(f"Error: Drone data path not found: {args.drone_base_path}")
        return

    if not Path(args.annotations_json).exists():
        print(f"Error: Annotations file not found: {args.annotations_json}")
        return

    converter = Drone3DNavigationDataConverter(args.drone_targets, args.drone_base_path, args.annotations_json, args.output_dir)
    train_path, val_path = converter.convert_to_llava_format(
        args.subset_train_size,
        args.subset_val_size,
        args.create_balanced
    )

    print("\n" + "=" * 70)
    print("✅ Conversion completed!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Default train: {train_path}")
    print(f"  Default val: {val_path}")
    print(f"  Images directory: {args.output_dir}/images")
    print(f"  Statistics: {args.output_dir}/drone_3d_conversion_statistics_full.json")


if __name__ == "__main__":
    main()
