#!/usr/bin/env python3
"""
R2R dataset-based navigation system test script.
Follows the standard pipeline using randomly selected trajectories from the R2R dataset.
"""

import json
import random
import shutil
from pathlib import Path
import time
import argparse
import torch

from navigation_system import test_navigation_system_siglip_aligned
from utils import setup_environment, get_best_gpu, get_safe_output_dir
from llava_navigation import LLaVANavigationModel


class R2RDatasetHandler:
    """R2R dataset handler."""
    
    def __init__(self, r2r_base_path: str):
        self.r2r_base_path = Path(r2r_base_path)
        self.annotations_path = self.r2r_base_path / "annotations_v1-3.json"
        self.images_base_path = self.r2r_base_path / "R2R_images_v1-3"
        
    def load_annotations(self):
        """Load R2R annotations."""
        try:
            with open(self.annotations_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            print(f"Loaded {len(annotations)} trajectories from R2R annotations")
            return annotations
        except Exception as e:
            print(f"Failed to load R2R annotations: {e}")
            return None
    
    def select_random_trajectory(self, annotations, min_path_length=10, max_path_length=50):
        """Select random trajectory with optimized path length range."""
        if not annotations:
            return None
            
        valid_trajectories = []
        for traj in annotations:
            if ('actions' in traj and 'instructions' in traj and 'video' in traj and
                min_path_length <= len(traj['actions']) <= max_path_length):
                valid_trajectories.append(traj)
        
        if not valid_trajectories:
            print(f"No valid trajectories found with path length {min_path_length}-{max_path_length}")
            return None
            
        selected = random.choice(valid_trajectories)
        print(f"Selected trajectory:")
        print(f"   ID: {selected.get('id')}")
        print(f"   Video path: {selected.get('video')}")
        print(f"   Action sequence length: {len(selected.get('actions', []))}")
        print(f"   Instruction: {selected['instructions'][0][:80]}...")
        
        return selected

    def select_trajectory_by_id(self, annotations, trajectory_id, min_path_length=10, max_path_length=50):
        """Select specific trajectory by ID."""
        if not annotations:
            return None

        target_trajectory = None
        for traj in annotations:
            if traj.get('id') == trajectory_id:
                target_trajectory = traj
                break

        if target_trajectory is None:
            print(f"Trajectory with ID '{trajectory_id}' not found")
            return None

        if not ('actions' in target_trajectory and 'instructions' in target_trajectory and 'video' in target_trajectory):
            print(f"Trajectory {trajectory_id} missing required fields")
            return None

        action_length = len(target_trajectory.get('actions', []))
        if not (min_path_length <= action_length <= max_path_length):
            print(f"Trajectory {trajectory_id} path length {action_length} not in range {min_path_length}-{max_path_length}")
            return None

        print(f"Selected trajectory by ID:")
        print(f"   ID: {target_trajectory.get('id')}")
        print(f"   Video path: {target_trajectory.get('video')}")
        print(f"   Action sequence length: {action_length}")
        print(f"   Instruction: {target_trajectory['instructions'][0][:80]}...")

        return target_trajectory

    def get_image_sequence(self, trajectory):
        """Get image sequence corresponding to trajectory."""
        video_path = trajectory.get('video')
        if not video_path:
            return None, None
            
        image_dir = self.images_base_path / video_path / "rgb"
        
        if not image_dir.exists():
            print(f"Image directory not found: {image_dir}")
            return None, None
            
        image_files = sorted(list(image_dir.glob("*.jpg")))
        
        if not image_files:
            print(f"No images found in: {image_dir}")
            return None, None
            
        print(f"Found {len(image_files)} images in trajectory")
        return [str(img) for img in image_files], trajectory['instructions'][0]
    
    def prepare_test_dataset(self, trajectory, target_dir):
        """Prepare test dataset by copying images to specified directory."""
        image_sequence, instruction = self.get_image_sequence(trajectory)
        
        if not image_sequence:
            return None, None
            
        test_dataset_dir = Path(target_dir) / "r2r_test_dataset"
        test_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        copied_images = []
        for i, img_path in enumerate(image_sequence):
            src_path = Path(img_path)
            dst_path = test_dataset_dir / src_path.name
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_images.append(str(dst_path))
            except Exception as e:
                print(f"Failed to copy {src_path}: {e}")
                continue
        
        print(f"Prepared test dataset with {len(copied_images)} images in {test_dataset_dir}")
        
        trajectory_info = {
            'trajectory_id': trajectory.get('id'),
            'instruction': instruction,
            'original_video_path': trajectory.get('video'),
            'action_sequence_length': len(trajectory.get('actions', [])),
            'image_count': len(copied_images),
            'test_dataset_path': str(test_dataset_dir)
        }
        
        info_path = test_dataset_dir / "trajectory_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_info, f, indent=2, ensure_ascii=False)
        
        return str(test_dataset_dir), instruction


def test_navigation_system_siglip_aligned_with_scene_analysis(dataset_path: str,
                                                              fastsam_model_path: str = "FastSAM-s.pt",
                                                              max_images: int = 100,
                                                              sample_interval: int = 10,
                                                              use_fastsam: bool = True,
                                                              quality_threshold: float = 0.65,
                                                              preferred_gpu_id: int = None,
                                                              siglip_model_path: str = "google/siglip-so400m-patch14-384",
                                                              dinov2_model: str = None,
                                                              llava_model=None,
                                                              merge_temporal_window: int = 10):
    """Test navigation system with SigLIP-aligned features and FastSAM (enhanced: supports scene analysis)."""

    print(f"Testing Navigation System with FastSAM + Scene Analysis on R2R Dataset")
    print(f"Dataset: {dataset_path}")

    from utils import validate_dataset_path
    valid, image_paths = validate_dataset_path(dataset_path)
    if not valid:
        print(f"Invalid dataset: {dataset_path}")
        return None

    print(f"Found {len(image_paths)} images")

    from navigation_system import LandmarkNavigationSystem
    nav_system = LandmarkNavigationSystem(
        fastsam_model_path=fastsam_model_path,
        siglip_model_path=siglip_model_path,
        dinov2_model=dinov2_model,
        use_fastsam=use_fastsam,
        base_similarity_threshold=0.75,
        preferred_gpu_id=preferred_gpu_id,
        merge_temporal_window=merge_temporal_window
    )

    if llava_model is not None:
        nav_system.set_llava_model(llava_model)
        print("Scene analysis enabled with LLaVA-OneVision")

    image_paths = image_paths[:max_images]

    try:
        image_paths_str = [str(p) for p in image_paths]
        nav_system.process_image_sequence(
            image_paths_str,
            sample_interval=sample_interval,
            quality_threshold=quality_threshold
        )

        nav_system.save_system()

        return nav_system

    except Exception as e:
        print(f"Processing failed: {e}")
        return None


def dual_target_navigation_inference_test(nav_system, llava_nav_model, dataset_path, args, navigation_instruction):
    """Execute dual-target navigation inference test (R2R version: using real instructions)."""
    try:
        print(f"Navigation Instruction: '{navigation_instruction}'")

        print(f"Parsing instruction...")
        start_parse = time.time()
        landmark_sequence = llava_nav_model.parse_navigation_instruction(navigation_instruction)
        parse_time = time.time() - start_parse
        print(f"Extracted landmarks: {landmark_sequence} (Parse time: {parse_time:.3f}s)")

        if not landmark_sequence:
            print("Failed to extract landmark sequence")
            return

        print(f"Querying topological graph...")
        target_results_dict = {}
        for landmark_text in landmark_sequence:
            results = nav_system.query_landmarks(landmark_text, top_k=2, save_visualization=False)
            if results:
                target_results_dict[landmark_text] = results
                print(f"   '{landmark_text}': Best match: {results[0][1]:.3f}")

        if not target_results_dict:
            print("No landmark targets found in topological graph")
            return

        completed_targets = []
        inference_completed = False

        for current_target_index, current_target_text in enumerate(landmark_sequence):
            if current_target_text not in target_results_dict:
                print(f"Target '{current_target_text}' not found in topological graph")
                break

            current_target_results = target_results_dict[current_target_text]
            target_landmark_id, target_similarity = current_target_results[0]

            print(f"\nProcessing target {current_target_index + 1}/{len(landmark_sequence)}: '{current_target_text}' -> L{target_landmark_id}")

            target_metadata = nav_system.topological_graph.landmark_metadata.get(target_landmark_id, {})
            target_scene_indoor_outdoor = target_metadata.get('scene_indoor_outdoor', 'unknown')
            target_scene_category = target_metadata.get('scene_category', 'unknown')
            if target_scene_indoor_outdoor != 'unknown':
                print(f"Target scene: {target_scene_indoor_outdoor}/{target_scene_category}")

            target_reference_path = nav_system.topological_graph.get_landmark_reference_image(target_landmark_id)
            if not target_reference_path:
                print(f"Target landmark L{target_landmark_id} has no reference image")
                break

            test_image_path = get_test_observation_image(dataset_path)
            if not test_image_path:
                print("No test observation image available")
                break

            print(f"Test observation: {Path(test_image_path).name}")

            if nav_system.processing_stats.get('scene_analysis_enabled', False):
                test_scene_info = nav_system.analyze_scene(test_image_path)
                if test_scene_info.get('analysis_success', False):
                    test_indoor_outdoor = test_scene_info.get('indoor_outdoor', 'unknown')
                    test_scene_category = test_scene_info.get('scene_category', 'unknown')
                    print(f"Current scene: {test_indoor_outdoor}/{test_scene_category}")

            print(f"Checking target visibility...")

            path_check_result = run_dual_target_navigation_pipeline(
                nav_system, llava_nav_model, test_image_path,
                navigation_instruction, landmark_sequence, current_target_text,
                target_landmark_id, target_reference_path
            )

            target_directly_visible = path_check_result and path_check_result.get('target_reached', False)

            if target_directly_visible:
                print(f"Target '{current_target_text}' (L{target_landmark_id}) is directly visible")
                completed_targets.append(current_target_text)
                continue
            else:
                print(f"Found first non-visible target: '{current_target_text}' (L{target_landmark_id})")
                print(f"Performing single navigation inference test...")

                test_iterations = args.test_iterations if hasattr(args, 'test_iterations') else 1
                control_latencies = []
                llava_only_latencies = []
                detailed_performance_data = []

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                control_start_time = time.time()

                control_command, llava_inference_time, detailed_perf = run_dual_target_navigation_pipeline(
                    nav_system, llava_nav_model, test_image_path,
                    navigation_instruction, landmark_sequence, current_target_text,
                    target_landmark_id, target_reference_path, return_timing=True
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                control_end_time = time.time()
                control_latency = control_end_time - control_start_time
                control_latencies.append(control_latency)
                llava_only_latencies.append(llava_inference_time)
                detailed_performance_data.append(detailed_perf)

                print(f"   Single inference: {control_command.get('action', 'UNKNOWN')} ({control_latency:.3f}s)")

                if args.navigation_visualization:
                    print("Generating dual-target navigation path visualization...")
                    try:
                        landmarks, _ = nav_system.landmark_extractor.extract_landmarks(test_image_path)

                        if landmarks:
                            matched_landmarks = nav_system.topological_graph.match_image_landmarks_without_adding(landmarks)

                            if matched_landmarks:
                                best_start_landmark, complete_navigation_path = nav_system.topological_graph.find_shortest_path_from_current_to_target(
                                    matched_landmarks, target_landmark_id
                                )

                                if best_start_landmark is not None and complete_navigation_path:
                                    current_landmarks = [best_start_landmark]
                                    nav_system.visualize_navigation_path(
                                        navigation_instruction=f"{navigation_instruction} (R2R Dataset + Dual-Target Strategy + Scene Aware)",
                                        target_landmark_id=target_landmark_id,
                                        target_similarity=target_similarity,
                                        current_image_path=test_image_path,
                                        current_landmarks=current_landmarks,
                                        navigation_path=complete_navigation_path
                                    )
                                else:
                                    print(f"   No path exists")
                            else:
                                print(f"   No matched landmarks found")

                    except Exception as viz_error:
                        print(f"   Navigation visualization failed: {viz_error}")

                if control_latencies and detailed_performance_data:
                    avg_control_latency = sum(control_latencies) / len(control_latencies)
                    control_frequency = 1 / avg_control_latency if avg_control_latency > 0 else 0
                    avg_llava_latency = sum(llava_only_latencies) / len(llava_only_latencies)

                    avg_fastsam_time = sum(perf['fastsam_time'] for perf in detailed_performance_data) / len(detailed_performance_data)
                    avg_siglip_time = sum(perf['siglip_time'] for perf in detailed_performance_data) / len(detailed_performance_data)
                    avg_dinov2_time = sum(perf['dinov2_time'] for perf in detailed_performance_data) / len(detailed_performance_data)
                    avg_dual_target_matching_time = sum(perf['dual_target_matching_time'] for perf in detailed_performance_data) / len(detailed_performance_data)
                    avg_scene_analysis_time = sum(perf.get('scene_analysis_time', 0.0) for perf in detailed_performance_data) / len(detailed_performance_data)

                    print(f"Performance Summary:")
                    print(f"   Average control latency: {avg_control_latency:.3f}s")
                    print(f"   Average LLaVA inference: {avg_llava_latency:.3f}s")
                    print(f"   Control frequency: {control_frequency:.1f}Hz")
                    print(f"   Real-time performance: {'Yes' if avg_control_latency < 0.333 else 'No'}")

                else:
                    avg_control_latency = -1.0
                    control_frequency = -1.0
                    avg_llava_latency = -1.0
                    avg_fastsam_time = -1.0
                    avg_siglip_time = -1.0
                    avg_dinov2_time = -1.0
                    avg_dual_target_matching_time = -1.0
                    avg_scene_analysis_time = -1.0
                    control_command = {"action": "HOVER", "reason": "Test failed", "confidence": 0.0}

                save_r2r_dual_target_results(
                    nav_system, navigation_instruction, landmark_sequence, current_target_text,
                    target_landmark_id, target_similarity, test_image_path, target_reference_path,
                    control_command, avg_control_latency, control_frequency, parse_time,
                    avg_fastsam_time, avg_siglip_time, avg_dinov2_time, avg_dual_target_matching_time,
                    avg_llava_latency, avg_scene_analysis_time, args, detailed_performance_data,
                    target_scene_indoor_outdoor, target_scene_category, completed_targets
                )

                inference_completed = True
                print(f"Inference completed for first non-visible target: '{current_target_text}'")
                break

        if inference_completed:
            print_final_navigation_summary_modified(navigation_instruction, landmark_sequence, completed_targets, True)
        else:
            print_final_navigation_summary_modified(navigation_instruction, landmark_sequence, completed_targets, False)

    except Exception as e:
        print(f"Dual-target navigation inference failed: {e}")
        import traceback
        traceback.print_exc()


def run_dual_target_navigation_pipeline(nav_system, llava_nav_model, test_image_path,
                                        navigation_instruction, landmark_sequence, current_target_text,
                                        target_landmark_id, target_reference_path, return_timing=False):
    """Run dual-target navigation pipeline (modified: supports target reached detection)."""
    detailed_perf = {
        'fastsam_time': 0.0,
        'siglip_time': 0.0,
        'dinov2_time': 0.0,
        'quality_computation_time': 0.0,
        'filtering_time': 0.0,
        'image_saving_time': 0.0,
        'dual_target_matching_time': 0.0,
        'scene_analysis_time': 0.0
    }

    landmarks, extraction_timing = nav_system.landmark_extractor.extract_landmarks(test_image_path)

    detailed_perf['fastsam_time'] = extraction_timing['fastsam_segmentation_time']
    detailed_perf['siglip_time'] = extraction_timing['siglip_feature_time']
    detailed_perf['dinov2_time'] = extraction_timing['dinov2_feature_time']
    detailed_perf['quality_computation_time'] = extraction_timing['quality_computation_time']
    detailed_perf['filtering_time'] = extraction_timing['filtering_time']
    detailed_perf['image_saving_time'] = extraction_timing['image_saving_time']

    visible_landmark_id = None
    target_reached = False

    if landmarks:
        dual_target_start = time.time()

        matched_landmarks = nav_system.topological_graph.match_image_landmarks_without_adding(landmarks)

        if matched_landmarks:
            best_start_landmark, shortest_path = nav_system.topological_graph.find_shortest_path_from_current_to_target(
                matched_landmarks, target_landmark_id
            )

            if best_start_landmark is not None and shortest_path:
                if len(shortest_path) == 1:
                    target_reached = True
                    print(f"Target reached: L{target_landmark_id} is directly visible in current view")

                    if not return_timing:
                        return {"target_reached": True, "reached_landmark_id": target_landmark_id}

                elif len(shortest_path) >= 2:
                    visible_landmark_id = shortest_path[0]
                    print(f"Dual-target strategy: Visible guide L{visible_landmark_id} -> Final target L{target_landmark_id}")
                else:
                    visible_landmark_id = None

        dual_target_end = time.time()
        detailed_perf['dual_target_matching_time'] = dual_target_end - dual_target_start

    if target_reached and return_timing:
        control_command = {"action": "TARGET_REACHED", "reason": f"Target L{target_landmark_id} reached",
                           "confidence": 1.0}
        llava_inference_time = 0.0
        return control_command, llava_inference_time, detailed_perf

    if target_reached:
        return {"target_reached": True, "reached_landmark_id": target_landmark_id}

    llava_start = time.time()
    control_command = llava_nav_model.generate_dual_target_navigation_command(
        full_instruction=navigation_instruction,
        landmark_sequence=landmark_sequence,
        current_target=current_target_text,
        current_image_path=test_image_path,
        visible_landmark_id=visible_landmark_id,
        target_landmark_id=target_landmark_id
    )
    llava_inference_time = time.time() - llava_start

    if return_timing:
        return control_command, llava_inference_time, detailed_perf
    else:
        return control_command


def get_test_observation_image(dataset_path):
    """Get observation image for testing."""
    try:
        dataset_path = Path(dataset_path)
        image_paths = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

        for ext in image_extensions:
            image_paths.extend(list(dataset_path.glob(f"**/{ext}")))
            image_paths.extend(list(dataset_path.glob(f"**/{ext.upper()}")))

        filtered_paths = [p for p in image_paths if "checkpoint" not in p.stem.lower()]
        filtered_paths.sort()

        if filtered_paths:
            return str(filtered_paths[0])
        else:
            return None
    except:
        return None


def save_r2r_dual_target_results(nav_system, instruction, landmark_sequence, current_target,
                                 target_landmark_id, target_similarity, test_image_path, target_reference_path,
                                 control_command, avg_control_latency, control_frequency, parse_time,
                                 avg_fastsam_time, avg_siglip_time, avg_dinov2_time, avg_dual_target_matching_time,
                                 avg_llava_latency, avg_scene_analysis_time, args, detailed_performance_data,
                                 target_scene_indoor_outdoor, target_scene_category, completed_targets=None):
    """Save R2R dual-target navigation results."""
    try:
        from utils import get_safe_output_dir
        import json
        from datetime import datetime

        output_dir = nav_system.landmark_extractor.session_dir.parent

        results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'r2r_dual_target_navigation_with_real_instructions',
            'dataset_source': 'R2R_v1-3',
            'model_used': f"llava-onevision-qwen2-{args.model_size}-ov",
            'navigation_strategy': 'dual_target_with_r2r_instructions',
            'scene_analysis_enabled': nav_system.processing_stats.get('scene_analysis_enabled', False),
            'navigation_instruction': instruction,
            'landmark_sequence': landmark_sequence,
            'completed_targets': completed_targets or [],
            'current_target': current_target,
            'target_landmark_id': target_landmark_id,
            'target_similarity': target_similarity,
            'target_scene_info': {
                'indoor_outdoor': target_scene_indoor_outdoor,
                'scene_category': target_scene_category
            },
            'test_image_path': test_image_path,
            'target_reference_path': target_reference_path,
            'control_command': control_command,
            'performance_metrics': {
                'control_average_latency_sec': avg_control_latency,
                'control_frequency_hz': control_frequency,
                'text_parsing_time_sec': parse_time,
                'fastsam_segmentation_sec': avg_fastsam_time,
                'siglip_features_sec': avg_siglip_time,
                'dinov2_features_sec': avg_dinov2_time,
                'dual_target_matching_sec': avg_dual_target_matching_time,
                'scene_analysis_sec': avg_scene_analysis_time,
                'llava_dual_target_inference_sec': avg_llava_latency,
                'realtime_target_achieved': avg_control_latency < 0.333,
                'speed_improvement_needed': avg_control_latency / 0.333,
            },
            'scene_analysis_stats': {
                'success_rate': nav_system.processing_stats.get('scene_analysis_success_rate', 0.0),
                'total_landmarks': len(nav_system.topological_graph.landmark_features),
                'indoor_landmarks': nav_system.topological_graph.stats['scene_analysis_stats']['indoor_landmarks'],
                'outdoor_landmarks': nav_system.topological_graph.stats['scene_analysis_stats']['outdoor_landmarks'],
                'unknown_scene_landmarks': nav_system.topological_graph.stats['scene_analysis_stats']['unknown_scene_landmarks'],
            },
            'r2r_dataset_innovation': {
                'real_world_instructions': True,
                'human_annotated_trajectories': True,
                'visual_semantic_grounding': True,
                'multi_modal_reasoning': True,
                'scene_aware_navigation': True
            },
            'dual_target_innovation': {
                'beyond_visual_servoing': True,
                'semantic_spatial_reasoning': True,
                'multi_landmark_context': True,
                'enhanced_exploration_capability': True,
                'r2r_instruction_grounding': True
            },
            'raw_performance_data': detailed_performance_data,
        }

        results_path = output_dir / f"r2r_dual_target_results_{args.model_size}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"R2R test results saved to: {results_path}")

    except Exception as e:
        print(f"Failed to save R2R dual-target results: {e}")


def print_final_navigation_summary_modified(navigation_instruction, landmark_sequence, completed_targets,
                                            inference_completed):
    """Print modified final navigation summary."""
    print(f"\n" + "=" * 80)
    print("FINAL R2R NAVIGATION SUMMARY (SINGLE INFERENCE MODE)")
    print("=" * 80)
    print(f"R2R Instruction: {navigation_instruction}")
    print(f"Planned Sequence: {' -> '.join(landmark_sequence)}")
    print(f"Visible Targets: {' -> '.join(completed_targets)}")

    if inference_completed:
        remaining = [target for target in landmark_sequence if target not in completed_targets]
        if remaining:
            print(f"First Non-Visible Target: {remaining[0]} (INFERENCE COMPLETED)")
        print("STATUS: SINGLE INFERENCE MODE - COMPLETED")
    else:
        print("STATUS: ALL TARGETS WERE VISIBLE OR ERROR OCCURRED")

    print(f"Strategy: R2R Real Instructions -> Find first non-visible target -> Single inference -> End")
    print("=" * 80)


def main(args):
    """Main execution function (R2R version)."""
    print("Starting R2R Navigation System with FastSAM + Scene Analysis + Dual-Target Navigation...")

    setup_environment()

    if args.model_size == "0.5b":
        llava_model_path = "/root/autodl-tmp/model/transformers/models--lmms-lab--llava-onevision-qwen2-0.5b-ov"
        print(f"Using small model: llava-onevision-qwen2-0.5b-ov")
    else:
        llava_model_path = "/root/autodl-tmp/model/transformers/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"
        print(f"Using large model: llava-onevision-qwen2-7b-ov")

    r2r_base_path = "/root/autodl-tmp/project/StreamVLN/R2R"
    fastsam_model_path = "FastSAM-x.pt"
    siglip_model_path = "google/siglip-so400m-patch14-384"
    dinov2_model_path = "/root/autodl-tmp/model/transformers/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415"

    gpu_id, gpu_name, gpu_memory = get_best_gpu()

    print("Processing R2R dataset...")
    try:
        r2r_handler = R2RDatasetHandler(r2r_base_path)
        
        annotations = r2r_handler.load_annotations()
        if not annotations:
            print("Failed to load R2R annotations")
            return
        
        if hasattr(args, 'trajectory_id') and args.trajectory_id:
            trajectory = r2r_handler.select_trajectory_by_id(annotations, args.trajectory_id)
        else:
            trajectory = r2r_handler.select_random_trajectory(annotations)

        if not trajectory:
            print("Failed to select trajectory")
            return
        
        output_dir = get_safe_output_dir()
        dataset_path, navigation_instruction = r2r_handler.prepare_test_dataset(trajectory, output_dir)
        if not dataset_path:
            print("Failed to prepare test dataset")
            return
            
        print(f"R2R test dataset prepared successfully")
        print(f"   Dataset path: {dataset_path}")
        print(f"   Navigation instruction: {navigation_instruction}")
        
    except Exception as e:
        print(f"R2R dataset processing failed: {e}")
        return

    print("Loading LLaVA-OneVision Model...")
    try:
        llava_nav_model = LLaVANavigationModel(
            model_path=llava_model_path,
            preferred_gpu_id=gpu_id,
            quantization_level=args.quantization
        )

        if args.enable_visualization:
            viz_dir = output_dir / "r2r_dual_target_inference_visualization"
            llava_nav_model.set_visualization_config(enable=True, output_dir=str(viz_dir))
        else:
            llava_nav_model.set_visualization_config(enable=False)

        print("LLaVA model loaded successfully")
    except Exception as e:
        print(f"LLaVA model loading failed: {e}")
        llava_nav_model = None

    print("Processing R2R images and building scene-aware topological graph...")
    try:
        nav_system = test_navigation_system_siglip_aligned_with_scene_analysis(
            dataset_path=dataset_path,
            fastsam_model_path=fastsam_model_path,
            max_images=50,
            sample_interval=1,
            use_fastsam=True,
            quality_threshold=0.65,
            preferred_gpu_id=gpu_id,
            siglip_model_path=siglip_model_path,
            dinov2_model=dinov2_model_path,
            llava_model=llava_nav_model
        )

        if nav_system:
            nav_system.landmark_extractor.configure_landmark_extraction(
                padding_ratio=0.12,
                use_rectangular=True,
                normalize_size=True,
                mark_bbox=False
            )

        if nav_system:
            print("Navigation system processing completed")

            if llava_nav_model:
                llava_nav_model.set_topological_graph(nav_system.topological_graph)

            stats = nav_system.processing_stats
            topo_stats = nav_system.topological_graph.stats

            print(f"\nSummary:")
            print(f"   Processed images: {stats['total_images_processed']}")
            print(f"   Total landmarks: {len(nav_system.topological_graph.landmark_features)}")
            print(f"   Landmarks w/ Images: {topo_stats['landmarks_with_images']}")
            print(f"   Processing time: {stats['total_processing_time']:.1f}s")

            if stats.get('scene_analysis_enabled', False):
                scene_success_rate = stats.get('scene_analysis_success_rate', 0.0)
                scene_stats = topo_stats.get('scene_analysis_stats', {})
                print(f"   Scene analysis success rate: {scene_success_rate:.1%}")
                print(f"   Indoor landmarks: {scene_stats.get('indoor_landmarks', 0)}")
                print(f"   Outdoor landmarks: {scene_stats.get('outdoor_landmarks', 0)}")

            print(f"Starting R2R Dual-Target Navigation Inference...")
            if llava_nav_model:
                dual_target_navigation_inference_test(nav_system, llava_nav_model, dataset_path, args, navigation_instruction)
            else:
                print("Cannot perform navigation inference: LLaVA model not available")

        else:
            print("Navigation system processing failed")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run R2R Dataset Dual-Target Navigation System")
    parser.add_argument('--model_size', type=str, default="7b", choices=['0.5b', '7b'],
                        help='Size of the LLaVA model to use.')
    parser.add_argument('--quantization', type=int, default=0, choices=[0, 4, 8],
                        help='Quantization level (0 for FP16).')
    parser.add_argument('--enable_visualization', action='store_true', default=True,
                        help='Enable detailed dual-target inference process visualization.')
    parser.add_argument('--navigation_visualization', action='store_true', default=True,
                        help='Enable dual-target navigation path visualization.')
    parser.add_argument('--test_iterations', type=int, default=1,
                        help='Number of dual-target control loop test iterations.')
    parser.add_argument('--trajectory_id', type=int, default=None,
                        help='Specific trajectory ID to test (if not provided, will select randomly)')
    args = parser.parse_args()

    main(args)
