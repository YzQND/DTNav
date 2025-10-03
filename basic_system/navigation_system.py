#!/usr/bin/env python3
"""
Optimized SigLIP feature-aligned navigation system module.
Supports landmark RGB image storage and multi-image navigation.
Based on geometric distance covisibility and path scoring.
Includes scene-aware topological graph construction.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import torch
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from utils import monitor_gpu_memory, get_best_gpu, set_gpu_device, validate_dataset_path, get_safe_output_dir
from landmark_extractor import LandmarkExtractor
from topological_graph import LandmarkTopologicalGraph


class LandmarkNavigationSystem:
    """Landmark navigation system using SigLIP-aligned features with RGB image storage and scene perception."""

    def __init__(self, fastsam_model_path: str = "FastSAM-s.pt",
                 siglip_model_path: str = "google/siglip-so400m-patch14-384",
                 dinov2_model: str = None,
                 use_fastsam: bool = True,
                 base_similarity_threshold: float = 0.75,
                 preferred_gpu_id: int = None,
                 merge_temporal_window: int = 10,
                 shared_landmark_extractor: LandmarkExtractor = None):

        print("Initializing Navigation System with FastSAM...")

        # Setup GPU
        if preferred_gpu_id is None:
            gpu_id, gpu_name, gpu_memory = get_best_gpu()
        else:
            gpu_id = preferred_gpu_id

        self.device = set_gpu_device(gpu_id)

        # Initialize landmark extractor
        if shared_landmark_extractor is not None:
            print("Using shared landmark extractor instance")
            self.landmark_extractor = shared_landmark_extractor
        else:
            print("Creating new landmark extractor instance")
            self.landmark_extractor = LandmarkExtractor(
                fastsam_model_path=fastsam_model_path,
                siglip_model_path=siglip_model_path,
                dinov2_model=dinov2_model,
                preferred_gpu_id=gpu_id
            )

        # Initialize topological graph
        self.topological_graph = LandmarkTopologicalGraph(
            base_similarity_threshold=base_similarity_threshold,
            siglip_encoder=self.landmark_extractor.siglip_encoder,
            preferred_gpu_id=gpu_id,
            merge_temporal_window=merge_temporal_window
        )

        # LLaVA model reference for scene analysis
        self.llava_model = None

        # Processing statistics
        self.processing_stats = {
            'total_images_processed': 0,
            'total_landmarks_extracted': 0,
            'total_processing_time': 0.0,
            'merging_success_rate': 0.0,
            'total_merges_attempted': 0,
            'total_merges_successful': 0,
            'feature_alignment_method': 'siglip_aligned',
            'image_storage_enabled': True,
            'segmentation_model': 'FastSAM',
            'covisibility_method': 'geometric_distance_based',
            'scene_analysis_enabled': False,
            'scene_analysis_success_rate': 0.0,
            'using_shared_models': shared_landmark_extractor is not None,
        }

        # Visualization data tracking
        self.visualization_data = {
            'image_landmarks': {},
            'merge_pairs': [],
            'image_paths': [],
        }

        print("Navigation System initialized successfully")

    def set_llava_model(self, llava_model):
        """Set LLaVA model reference for scene analysis."""
        self.llava_model = llava_model
        if llava_model is not None:
            self.processing_stats['scene_analysis_enabled'] = True
            print("LLaVA model configured for scene analysis")

    def analyze_scene(self, image_path: str) -> dict:
        """Analyze scene information using LLaVA model."""
        if self.llava_model is None:
            return {
                'indoor_outdoor': 'unknown',
                'scene_category': 'unknown',
                'confidence': 0.0,
                'analysis_success': False
            }

        try:
            from llava.constants import DEFAULT_IMAGE_TOKEN
            
            scene_prompt = f"""Analyze this image and determine:
1. Environment type: indoor or outdoor
2. Specific scene category (e.g., kitchen, corridor, garden, street, office, bedroom, living room, etc.)

{DEFAULT_IMAGE_TOKEN}

Reply in this exact format:
Environment: [indoor/outdoor]
Category: [specific scene name]"""

            conv = self.llava_model.prepare_conversation_template()
            if conv is None:
                return {
                    'indoor_outdoor': 'unknown',
                    'scene_category': 'unknown',
                    'confidence': 0.0,
                    'analysis_success': False
                }

            image_tensor, image_size = self.llava_model.process_image_for_model(image_path)
            if image_tensor is None:
                return {
                    'indoor_outdoor': 'unknown',
                    'scene_category': 'unknown',
                    'confidence': 0.0,
                    'analysis_success': False
                }

            conv.append_message(conv.roles[0], scene_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            from llava.mm_utils import tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX
            input_ids = tokenizer_image_token(
                prompt, self.llava_model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.llava_model.device)

            with torch.inference_mode():
                output_ids = self.llava_model.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=32,
                    use_cache=True,
                    eos_token_id=self.llava_model.tokenizer.eos_token_id,
                )

            full_output = self.llava_model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            scene_info = self._parse_scene_analysis(full_output)
            scene_info['analysis_success'] = True

            return scene_info

        except Exception as e:
            print(f"Scene analysis failed for {Path(image_path).name}: {e}")
            return {
                'indoor_outdoor': 'unknown',
                'scene_category': 'unknown',
                'confidence': 0.0,
                'analysis_success': False
            }

    def _parse_scene_analysis(self, llava_output: str) -> dict:
        """Parse LLaVA scene analysis output."""
        try:
            clean_output = llava_output.replace("<|im_end|>", "").strip().lower()

            indoor_outdoor = 'unknown'
            if 'environment:' in clean_output:
                env_line = [line for line in clean_output.split('\n') if 'environment:' in line]
                if env_line:
                    env_text = env_line[0].split('environment:')[1].strip()
                    if 'indoor' in env_text:
                        indoor_outdoor = 'indoor'
                    elif 'outdoor' in env_text:
                        indoor_outdoor = 'outdoor'
            else:
                if 'indoor' in clean_output:
                    indoor_outdoor = 'indoor'
                elif 'outdoor' in clean_output:
                    indoor_outdoor = 'outdoor'

            scene_category = 'unknown'
            if 'category:' in clean_output:
                cat_line = [line for line in clean_output.split('\n') if 'category:' in line]
                if cat_line:
                    category_text = cat_line[0].split('category:')[1].strip()
                    scene_category = category_text.replace('.', '').replace(',', '').strip()

            confidence = 0.5
            if indoor_outdoor != 'unknown':
                confidence += 0.3
            if scene_category != 'unknown' and len(scene_category) > 2:
                confidence += 0.2

            return {
                'indoor_outdoor': indoor_outdoor,
                'scene_category': scene_category,
                'confidence': min(1.0, confidence),
                'raw_output': clean_output[:100]
            }

        except Exception:
            return {
                'indoor_outdoor': 'unknown',
                'scene_category': 'unknown',
                'confidence': 0.0,
                'raw_output': llava_output[:100] if llava_output else ""
            }

    def query_landmarks(self, instruction: str, top_k: int = 5, save_visualization: bool = True):
        """Query landmarks using natural language."""
        results = self.topological_graph.query_landmarks_by_language(instruction, top_k)

        if save_visualization and results:
            self._visualize_query_results(instruction, results, top_k=min(3, len(results)))

        return results

    def get_navigation_path(self, current_image_path: str, target_instruction: str):
        """Get navigation path to target."""
        return self.topological_graph.get_landmark_path_to_target(current_image_path, target_instruction)

    def get_landmarks_with_images(self, image_path: str = None):
        """Get landmark information with reference images."""
        if image_path:
            return self.topological_graph.get_current_view_landmarks_with_images(image_path)
        else:
            landmarks_with_images = []
            for landmark_id, metadata in self.topological_graph.landmark_metadata.items():
                reference_path = self.topological_graph.get_landmark_reference_image(landmark_id)
                if reference_path:
                    landmarks_with_images.append((landmark_id, reference_path))
            return landmarks_with_images

    def find_best_current_landmark_via_path(self, extracted_landmarks: List[dict], target_landmark_id: int) -> tuple:
        """Find best current landmark based on topological graph path."""
        if not extracted_landmarks:
            return None, 0.0

        matched_landmarks = self.topological_graph.match_image_landmarks_without_adding(extracted_landmarks)

        if not matched_landmarks:
            return None, 0.0

        best_start_landmark, shortest_path = self.topological_graph.find_shortest_path_from_current_to_target(
            matched_landmarks, target_landmark_id
        )

        if best_start_landmark is not None and shortest_path:
            if shortest_path:
                try:
                    import networkx as nx
                    geometric_distance = nx.shortest_path_length(
                        self.topological_graph.graph,
                        best_start_landmark,
                        target_landmark_id,
                        weight='geometric_distance'
                    )

                    similarity_score = 0.0
                    for landmark_id, sim_score in matched_landmarks:
                        if landmark_id == best_start_landmark:
                            similarity_score = sim_score
                            break

                    distance_factor = max(0.1, 1.0 - min(1.0, geometric_distance / 2.0))
                    similarity_factor = similarity_score
                    relevance_score = distance_factor * 0.7 + similarity_factor * 0.3

                    return best_start_landmark, relevance_score

                except Exception:
                    relevance_score = 1.0 / len(shortest_path) if len(shortest_path) > 0 else 0.0
                    return best_start_landmark, relevance_score

        return None, 0.0

    def visualize_navigation_path(self, navigation_instruction: str, target_landmark_id: int,
                                  target_similarity: float, current_image_path: str,
                                  current_landmarks: List[int], navigation_path: List[int]):
        """Enhanced navigation path visualization."""
        try:
            viz_dir = get_safe_output_dir() / "navigation_path_visualization"
            viz_dir.mkdir(exist_ok=True)

            if navigation_path and len(navigation_path) > 1:
                self._visualize_complete_navigation_path_enhanced(
                    viz_dir, navigation_instruction, target_landmark_id, target_similarity,
                    current_image_path, current_landmarks, navigation_path
                )
            elif target_landmark_id in current_landmarks:
                self._visualize_direct_navigation(
                    viz_dir, navigation_instruction, target_landmark_id, target_similarity,
                    current_image_path, current_landmarks
                )
            else:
                self._visualize_no_path_found(
                    viz_dir, navigation_instruction, target_landmark_id, target_similarity,
                    current_image_path, current_landmarks
                )

        except Exception as e:
            print(f"Navigation visualization failed: {e}")

    def _visualize_complete_navigation_path_enhanced(self, viz_dir, navigation_instruction, target_landmark_id,
                                                     target_similarity, current_image_path, current_landmarks,
                                                     navigation_path):
        """Visualize complete navigation path with enhanced details."""
        path_analysis = self._analyze_navigation_path(navigation_path)

        path_length = len(navigation_path)
        cols = min(4, path_length)
        rows = max(2, (path_length + cols - 1) // cols)

        fig = plt.figure(figsize=(6 * cols, 8 * rows + 4))

        fig.suptitle(f'Enhanced Multi-Image Navigation Path\n'
                     f'Instruction: "{navigation_instruction}"\n'
                     f'Target: L{target_landmark_id} (similarity: {target_similarity:.3f})',
                     fontsize=18, fontweight='bold', y=0.98)

        for i, landmark_id in enumerate(navigation_path):
            ax = plt.subplot(rows + 1, cols, i + 1)
            node_role = self._get_node_role(i, len(navigation_path))

            reference_image_path = self.topological_graph.get_landmark_reference_image(landmark_id)

            if reference_image_path and Path(reference_image_path).exists():
                try:
                    landmark_image = cv2.imread(reference_image_path)
                    if landmark_image is not None:
                        landmark_rgb = cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB)
                        ax.imshow(landmark_rgb)

                        color, label = self._get_enhanced_node_style(node_role, landmark_id, i)

                        h, w = landmark_rgb.shape[:2]
                        rect = Rectangle((0, 0), w - 1, h - 1, linewidth=6, edgecolor=color, facecolor='none',
                                         alpha=0.9)
                        ax.add_patch(rect)

                        enhanced_label = self._create_enhanced_node_label(
                            node_role, landmark_id, i, path_analysis, current_image_path
                        )

                        ax.text(10, 30, enhanced_label,
                                bbox=dict(boxstyle='round,pad=0.8', facecolor=color, alpha=0.9),
                                fontsize=11, fontweight='bold', color='white')

                        ax.set_title(f'{node_role["title"]} - L{landmark_id}\n{Path(reference_image_path).name}\n'
                                     f'Step {i + 1}/{len(navigation_path)}',
                                     fontsize=12, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, f'Failed to load reference image',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'L{landmark_id} - Step {i + 1}')

                except Exception:
                    ax.text(0.5, 0.5, f'Error loading reference image',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'L{landmark_id} - Step {i + 1}')
            else:
                metadata = self.topological_graph.landmark_metadata.get(landmark_id)
                if metadata and metadata['occurrences']:
                    best_occurrence = max(metadata['occurrences'], key=lambda x: x['quality_score'])
                    self._draw_landmark_from_original_enhanced(ax, best_occurrence, landmark_id, i,
                                                               len(navigation_path), node_role)
                else:
                    ax.text(0.5, 0.5, f'L{landmark_id}\nNo image data',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'L{landmark_id} - Step {i + 1}')

            ax.axis('off')

        info_ax = plt.subplot(rows + 1, 1, rows + 1)
        info_ax.axis('off')

        enhanced_path_info = self._create_enhanced_path_info_text(
            navigation_instruction, target_landmark_id, target_similarity,
            current_image_path, navigation_path, path_analysis
        )

        info_ax.text(0.05, 0.95, enhanced_path_info, transform=info_ax.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        safe_instruction = "".join(c for c in navigation_instruction if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_instruction = safe_instruction.replace(' ', '_')
        output_path = viz_dir / f"enhanced_navigation_path_{safe_instruction}_L{target_landmark_id}_{len(navigation_path)}steps.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    def _analyze_navigation_path(self, navigation_path: List[int]) -> dict:
        """Analyze detailed navigation path information."""
        analysis = {
            'total_distance': 0.0,
            'segment_distances': [],
            'path_complexity': 'Simple',
            'landmarks_with_images': 0,
            'step_details': []
        }

        try:
            import networkx as nx

            for i in range(len(navigation_path) - 1):
                current_id = navigation_path[i]
                next_id = navigation_path[i + 1]

                if self.topological_graph.graph.has_edge(current_id, next_id):
                    edge_data = self.topological_graph.graph.get_edge_data(current_id, next_id)
                    segment_distance = edge_data.get('geometric_distance', 0.0)
                    analysis['total_distance'] += segment_distance
                    analysis['segment_distances'].append(segment_distance)

                    analysis['step_details'].append({
                        'from': current_id,
                        'to': next_id,
                        'distance': segment_distance,
                        'step': i + 1
                    })

            for landmark_id in navigation_path:
                if self.topological_graph.get_landmark_reference_image(landmark_id):
                    analysis['landmarks_with_images'] += 1

            if len(navigation_path) <= 2:
                analysis['path_complexity'] = 'Direct'
            elif len(navigation_path) <= 4:
                analysis['path_complexity'] = 'Simple'
            elif len(navigation_path) <= 6:
                analysis['path_complexity'] = 'Moderate'
            else:
                analysis['path_complexity'] = 'Complex'

        except Exception:
            pass

        return analysis

    def _get_node_role(self, index: int, total_length: int) -> dict:
        """Get node role in path."""
        if index == 0:
            return {'type': 'start', 'title': 'START', 'description': 'Current Position'}
        elif index == total_length - 1:
            return {'type': 'target', 'title': 'TARGET', 'description': 'Final Destination'}
        else:
            return {'type': 'waypoint', 'title': f'WAYPOINT {index}', 'description': f'Intermediate Step {index}'}

    def _get_enhanced_node_style(self, node_role: dict, landmark_id: int, index: int) -> tuple:
        """Get enhanced node style."""
        if node_role['type'] == 'start':
            return 'lime', 'START'
        elif node_role['type'] == 'target':
            return 'red', 'TARGET'
        else:
            return 'orange', f'STEP {index + 1}'

    def _create_enhanced_node_label(self, node_role: dict, landmark_id: int, index: int,
                                    path_analysis: dict, current_image_path: str) -> str:
        """Create enhanced node label."""
        base_label = f"{node_role['title']}\nL{landmark_id}"
        base_label += f"\n{node_role['description']}"

        if index > 0 and index - 1 < len(path_analysis['segment_distances']):
            segment_dist = path_analysis['segment_distances'][index - 1]
            base_label += f"\nDist: {segment_dist:.3f}"

        has_image = self.topological_graph.get_landmark_reference_image(landmark_id) is not None
        base_label += f"\n{'Image: Yes' if has_image else 'Image: No'}"

        return base_label

    def _create_enhanced_path_info_text(self, navigation_instruction: str, target_landmark_id: int,
                                        target_similarity: float, current_image_path: str,
                                        navigation_path: List[int], path_analysis: dict) -> str:
        """Create enhanced path information text."""
        info_text = f"""Enhanced Navigation Path Analysis

Mission Summary:
   • Instruction: "{navigation_instruction}"
   • Target: L{target_landmark_id} (similarity: {target_similarity:.3f})
   • Current View: {Path(current_image_path).name}
   • Path Complexity: {path_analysis['path_complexity']}

Path Details:
   • Total Steps: {len(navigation_path)}
   • Total Distance: {path_analysis['total_distance']:.3f}
   • Landmarks with Images: {path_analysis['landmarks_with_images']}/{len(navigation_path)}

Route Sequence:
   {' -> '.join([f'L{lid}' for lid in navigation_path])}

Technical Info:
   • Segmentation: FastSAM
   • Feature Alignment: SigLIP
   • Covisibility: Geometric Distance
   • Path Algorithm: Dijkstra (geometric weighted)
   • Scene Analysis: {'Enabled' if self.processing_stats['scene_analysis_enabled'] else 'Disabled'}
"""
        return info_text

    def _draw_landmark_from_original_enhanced(self, ax, occurrence, landmark_id, step_index, total_steps, node_role):
        """Draw landmark from original image (enhanced version)."""
        try:
            image_path = occurrence['image_path']
            bbox = occurrence['bbox']
            center = occurrence['center']
            quality = occurrence['quality_score']

            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image_rgb)

                color, _ = self._get_enhanced_node_style(node_role, landmark_id, step_index)

                rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=5, edgecolor=color, facecolor='none', alpha=0.9)
                ax.add_patch(rect)

                ax.plot(center[0], center[1], 'o', color=color, markersize=15,
                        markeredgecolor='white', markeredgewidth=3)

                enhanced_label = f"{node_role['title']}\nL{landmark_id}\nQ: {quality:.3f}"

                ax.annotate(enhanced_label, (bbox[0], bbox[1]),
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.8', facecolor=color, alpha=0.9),
                            fontsize=10, fontweight='bold', color='white')

                ax.set_title(f'{node_role["title"]} - L{landmark_id}\n{Path(image_path).name}\nStep {step_index + 1}/{total_steps}',
                             fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'Image not found\n{Path(image_path).name}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'L{landmark_id} - Step {step_index + 1}')
        except Exception:
            ax.text(0.5, 0.5, f'Error loading image',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'L{landmark_id} - Step {step_index + 1}')

    def _visualize_direct_navigation(self, viz_dir, navigation_instruction, target_landmark_id,
                                     target_similarity, current_image_path, current_landmarks):
        """Visualize direct navigation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        self._draw_current_view(ax1, current_image_path, current_landmarks, target_landmark_id)

        reference_path = self.topological_graph.get_landmark_reference_image(target_landmark_id)
        if reference_path and Path(reference_path).exists():
            try:
                target_image = cv2.imread(reference_path)
                if target_image is not None:
                    target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                    ax2.imshow(target_rgb)

                    h, w = target_rgb.shape[:2]
                    rect = Rectangle((0, 0), w - 1, h - 1, linewidth=6, edgecolor='red', facecolor='none', alpha=0.9)
                    ax2.add_patch(rect)

                    ax2.text(10, 30, f"TARGET L{target_landmark_id}\nSim: {target_similarity:.3f}",
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                             fontsize=12, fontweight='bold', color='white')

                ax2.set_title(f'Target Reference Image\nL{target_landmark_id}\n{Path(reference_path).name}',
                              fontsize=12, fontweight='bold')
            except Exception:
                ax2.text(0.5, 0.5, f'Error loading target reference',
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'Target Reference L{target_landmark_id}')
        else:
            ax2.text(0.5, 0.5, 'No reference image available for target',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Target L{target_landmark_id} (No Reference)')

        ax2.axis('off')

        fig.suptitle(f'Direct Navigation - Target in Current View\n'
                     f'Instruction: "{navigation_instruction}"\n'
                     f'Target: L{target_landmark_id} (similarity: {target_similarity:.3f})',
                     fontsize=14, fontweight='bold', y=0.95)

        plt.tight_layout()

        safe_instruction = "".join(c for c in navigation_instruction if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_instruction = safe_instruction.replace(' ', '_')
        output_path = viz_dir / f"direct_navigation_{safe_instruction}_L{target_landmark_id}.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    def _draw_current_view(self, ax, current_image_path, current_landmarks, target_landmark_id):
        """Draw current observation view."""
        image = cv2.imread(current_image_path)
        if image is None:
            ax.text(0.5, 0.5, 'Current image not found', ha='center', va='center', transform=ax.transAxes)
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

        colors = plt.cm.Set3(np.linspace(0, 1, len(current_landmarks)))

        for i, landmark_id in enumerate(current_landmarks):
            color = colors[i]
            metadata = self.topological_graph.landmark_metadata.get(landmark_id)
            if not metadata:
                continue

            current_occurrence = None
            for occ in metadata['occurrences']:
                if occ['image_path'] == current_image_path:
                    current_occurrence = occ
                    break

            if not current_occurrence:
                continue

            bbox = current_occurrence['bbox']
            center = current_occurrence['center']
            quality = current_occurrence['quality_score']
            has_image = self.topological_graph.get_landmark_reference_image(landmark_id) is not None

            if landmark_id == target_landmark_id:
                edge_color = 'red'
                linewidth = 6
                marker_color = 'red'
                marker_size = 16
                label_text = f"TARGET L{landmark_id}\nQ: {quality:.3f}\n{'Img' if has_image else 'No Img'}"
                label_color = 'red'
            else:
                edge_color = color
                linewidth = 3
                marker_color = color
                marker_size = 10
                label_text = f"L{landmark_id}\nQ: {quality:.3f}\n{'Img' if has_image else 'No Img'}"
                label_color = color

            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             linewidth=linewidth, edgecolor=edge_color, facecolor='none', alpha=0.9)
            ax.add_patch(rect)

            ax.plot(center[0], center[1], 'o', color=marker_color, markersize=marker_size,
                    markeredgecolor='white', markeredgewidth=2)

            ax.annotate(label_text, (bbox[0], bbox[1]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=label_color, alpha=0.8),
                        fontsize=10, fontweight='bold', color='white')

        ax.set_title(f'Current View\n{Path(current_image_path).name}\n'
                     f'Landmarks: {current_landmarks}',
                     fontsize=12, fontweight='bold')
        ax.axis('off')

    def _visualize_no_path_found(self, viz_dir, navigation_instruction, target_landmark_id,
                                 target_similarity, current_image_path, current_landmarks):
        """Visualize no path found scenario."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        self._draw_current_view(ax1, current_image_path, current_landmarks, target_landmark_id)

        reference_path = self.topological_graph.get_landmark_reference_image(target_landmark_id)
        if reference_path and Path(reference_path).exists():
            try:
                target_image = cv2.imread(reference_path)
                if target_image is not None:
                    target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                    ax2.imshow(target_rgb)

                    h, w = target_rgb.shape[:2]
                    rect = Rectangle((0, 0), w - 1, h - 1, linewidth=6, edgecolor='red', facecolor='none', alpha=0.9)
                    ax2.add_patch(rect)

                    ax2.text(10, 30, f"TARGET L{target_landmark_id}\nSim: {target_similarity:.3f}",
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                             fontsize=12, fontweight='bold', color='white')

                ax2.set_title(f'Target Reference Image\nL{target_landmark_id}\n{Path(reference_path).name}',
                              fontsize=12, fontweight='bold')
            except Exception:
                ax2.text(0.5, 0.5, f'Error loading target reference',
                         ha='center', va='center', transform=ax2.transAxes)
        else:
            target_metadata = self.topological_graph.landmark_metadata.get(target_landmark_id)
            if target_metadata and target_metadata['occurrences']:
                best_occurrence = max(target_metadata['occurrences'], key=lambda x: x['quality_score'])
                self._draw_landmark_from_original(ax2, best_occurrence, target_landmark_id, -1, 1)
            else:
                ax2.text(0.5, 0.5, 'No target image available',
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'Target L{target_landmark_id} (No Image)')

        ax2.axis('off')

        fig.suptitle(f'Navigation - No Path Found\nInstruction: "{navigation_instruction}"\n'
                     f'Target: L{target_landmark_id} (similarity: {target_similarity:.3f})\n'
                     f'No connection between current landmarks and target',
                     fontsize=14, fontweight='bold', y=0.95)

        plt.tight_layout()

        safe_instruction = "".join(c for c in navigation_instruction if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_instruction = safe_instruction.replace(' ', '_')
        output_path = viz_dir / f"no_path_found_{safe_instruction}_L{target_landmark_id}.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    def _draw_landmark_from_original(self, ax, occurrence, landmark_id, step_index, total_steps):
        """Draw landmark from original image (fallback method)."""
        try:
            image_path = occurrence['image_path']
            bbox = occurrence['bbox']
            center = occurrence['center']
            quality = occurrence['quality_score']

            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image_rgb)

                if step_index == 0:
                    color = 'lime'
                    label = 'START'
                elif step_index == total_steps - 1:
                    color = 'red'
                    label = 'TARGET'
                else:
                    color = 'orange'
                    label = f'STEP {step_index + 1}'

                rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=4, edgecolor=color, facecolor='none', alpha=0.9)
                ax.add_patch(rect)

                ax.plot(center[0], center[1], 'o', color=color, markersize=12,
                        markeredgecolor='white', markeredgewidth=3)

                label_text = f"{label}\nQ: {quality:.3f}"
                ax.annotate(label_text, (bbox[0], bbox[1]),
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8),
                            fontsize=10, fontweight='bold', color='white')

                ax.set_title(f'L{landmark_id} - Step {step_index + 1}\n{Path(image_path).name}',
                             fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'Image not found\n{Path(image_path).name}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'L{landmark_id} - Step {step_index + 1}')
        except Exception:
            ax.text(0.5, 0.5, f'Error loading image',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'L{landmark_id} - Step {step_index + 1}')

    def analyze_current_scene(self, image_path: str) -> dict:
        """Analyze current scene and return detailed landmark and spatial information."""
        scene_analysis = {
            'image_path': image_path,
            'landmarks_found': [],
            'landmarks_with_images': [],
            'spatial_layout': {},
            'dominant_landmarks': [],
            'scene_confidence': 0.0,
            'segmentation_model': 'FastSAM',
            'covisibility_method': 'geometric_distance_based',
        }

        current_landmarks = self.topological_graph.image_landmarks.get(image_path, [])

        if not current_landmarks:
            return scene_analysis

        landmark_details = []
        landmarks_with_images = []

        for landmark_id in current_landmarks:
            if landmark_id in self.topological_graph.landmark_metadata:
                metadata = self.topological_graph.landmark_metadata[landmark_id]
                has_image = self.topological_graph.get_landmark_reference_image(landmark_id) is not None

                detail = {
                    'id': landmark_id,
                    'center': metadata.get('representative_center', [0, 0]),
                    'bbox': metadata.get('representative_bbox', [0, 0, 0, 0]),
                    'area': metadata.get('average_area', 0),
                    'quality': metadata.get('average_quality', 0),
                    'confidence': metadata.get('average_quality', 0) * min(1.0,
                                                                           metadata.get('total_occurrences', 1) / 3.0),
                    'has_reference_image': has_image
                }
                landmark_details.append(detail)

                if has_image:
                    landmarks_with_images.append(landmark_id)

        landmark_details.sort(key=lambda x: x['confidence'], reverse=True)
        scene_analysis['landmarks_found'] = landmark_details
        scene_analysis['landmarks_with_images'] = landmarks_with_images

        scene_analysis['dominant_landmarks'] = landmark_details[:3]

        if landmark_details:
            avg_confidence = np.mean([ld['confidence'] for ld in landmark_details])
            landmark_count_factor = min(1.0, len(landmark_details) / 5.0)
            image_availability_factor = len(landmarks_with_images) / len(landmark_details) if landmark_details else 0
            scene_analysis['scene_confidence'] = avg_confidence * landmark_count_factor * (
                    0.5 + 0.5 * image_availability_factor)

        if len(landmark_details) >= 2:
            centers = [ld['center'] for ld in landmark_details]
            scene_analysis['spatial_layout'] = {
                'center_of_mass': [np.mean([c[0] for c in centers]), np.mean([c[1] for c in centers])],
                'spread': np.std([c[0] for c in centers]) + np.std([c[1] for c in centers]),
                'landmark_count': len(landmark_details),
                'image_coverage': len(landmarks_with_images) / len(landmark_details) if landmark_details else 0
            }

        return scene_analysis

    def process_image_sequence(self, image_paths: List[str],
                               timestamps: List[float] = None,
                               sample_interval: int = 10,
                               max_images: int = None,
                               quality_threshold: float = 0.65):
        """Process image sequence with visualization tracking and scene analysis."""
        if timestamps is None:
            timestamps = [i * 1.0 for i in range(len(image_paths))]

        if max_images:
            image_paths = image_paths[:max_images]
            timestamps = timestamps[:max_images]

        sampled_indices = list(range(0, len(image_paths), sample_interval))
        sampled_paths = [image_paths[i] for i in sampled_indices]
        sampled_timestamps = [timestamps[i] for i in sampled_indices]

        print(f"Processing {len(sampled_paths)} images with FastSAM...")
        if self.processing_stats['scene_analysis_enabled']:
            print(f"Scene analysis enabled with LLaVA")

        total_start_time = time.time()
        scene_analysis_success_count = 0

        for i, (image_path, timestamp) in enumerate(zip(sampled_paths, sampled_timestamps)):
            try:
                scene_info = None
                if self.processing_stats['scene_analysis_enabled']:
                    scene_info = self.analyze_scene(image_path)
                    if scene_info.get('analysis_success', False):
                        scene_analysis_success_count += 1
                        print(f"[{i + 1}/{len(sampled_paths)}] Scene: {scene_info['indoor_outdoor']}/{scene_info['scene_category']}")

                landmarks, extraction_timing = self.landmark_extractor.extract_landmarks(image_path)

                high_quality_landmarks = [
                    lm for lm in landmarks
                    if lm.get('quality_score', 0) >= quality_threshold
                ]

                print(f"[{i + 1}/{len(sampled_paths)}] {Path(image_path).name}: {len(high_quality_landmarks)} landmarks")

                self.visualization_data['image_landmarks'][image_path] = {
                    'landmarks': high_quality_landmarks,
                    'image_index': i,
                    'timestamp': timestamp,
                    'scene_info': scene_info
                }
                self.visualization_data['image_paths'].append(image_path)

                merge_results = None
                if high_quality_landmarks:
                    merge_results = self.topological_graph.add_image_landmarks(
                        image_path, high_quality_landmarks, timestamp, scene_info=scene_info
                    )

                if merge_results and 'merge_details' in merge_results:
                    self._track_merge_pairs(image_path, merge_results['merge_details'], i)

                if merge_results:
                    self._update_stats(landmarks, merge_results)

                if (i + 1) % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {Path(image_path).name}: {e}")
                continue

        total_processing_time = time.time() - total_start_time
        self.processing_stats['total_processing_time'] = total_processing_time

        if self.processing_stats['scene_analysis_enabled'] and len(sampled_paths) > 0:
            self.processing_stats['scene_analysis_success_rate'] = scene_analysis_success_count / len(sampled_paths)

        print(f"Processing completed in {total_processing_time:.1f}s")
        if self.processing_stats['scene_analysis_enabled']:
            print(f"Scene analysis success rate: {self.processing_stats['scene_analysis_success_rate']:.1%}")

        self.topological_graph.print_stats()

    def _track_merge_pairs(self, current_image_path, merge_details, image_index):
        """Track successful merge pairs for visualization."""
        for detail in merge_details:
            if detail['action'] == 'merged':
                merge_pair = {
                    'current_image': current_image_path,
                    'current_image_index': image_index,
                    'current_landmark_local_id': detail['local_id'],
                    'target_landmark_id': detail['target_id'],
                    'similarity_score': detail['similarity_score'],
                    'similarity_details': detail.get('similarity_details', {}),
                    'merge_timestamp': time.time()
                }

                target_metadata = self.topological_graph.landmark_metadata.get(detail['target_id'])
                if target_metadata and target_metadata['occurrences']:
                    first_occurrence = target_metadata['occurrences'][0]
                    merge_pair['target_image'] = first_occurrence['image_path']
                    merge_pair['target_bbox'] = first_occurrence['bbox']
                    merge_pair['target_center'] = first_occurrence['center']

                self.visualization_data['merge_pairs'].append(merge_pair)

    def _update_stats(self, landmarks, merge_results):
        """Update processing statistics."""
        self.processing_stats['total_images_processed'] += 1
        self.processing_stats['total_landmarks_extracted'] += len(landmarks)

        if isinstance(merge_results, dict):
            attempted = merge_results.get('attempted_merges', 0)
            successful = merge_results.get('successful_merges', 0)

            self.processing_stats['total_merges_attempted'] += attempted
            self.processing_stats['total_merges_successful'] += successful

            if self.processing_stats['total_merges_attempted'] > 0:
                self.processing_stats['merging_success_rate'] = (
                        self.processing_stats['total_merges_successful'] /
                        self.processing_stats['total_merges_attempted']
                )

    def save_system(self, save_dir: str = None):
        """Save system state and visualizations."""
        if save_dir is None:
            save_dir = get_safe_output_dir()
        save_dir = Path(save_dir)
        print(f"Saving results to {save_dir}")

        try:
            graph_path = save_dir / "topological_graph_siglip_fastsam_geometric_covisibility_scene_aware.json"
            self.topological_graph.save_graph(str(graph_path))

            self._save_landmark_visualizations(save_dir)
            self._save_merge_pair_visualizations(save_dir)
            self._save_graph_visualization(save_dir)
            self._save_system_info(save_dir)
            self._save_image_storage_info(save_dir)

            print(f"Results saved to: {save_dir}")

        except Exception as e:
            print(f"Save error: {e}")

    def _save_image_storage_info(self, save_dir):
        """Save image storage information."""
        try:
            total_landmarks = len(self.topological_graph.landmark_features)
            landmarks_with_images = self.topological_graph.stats.get('landmarks_with_images', 0)
            total_images = self.topological_graph.stats.get('total_landmark_images', 0)

            landmark_image_info = {}
            for landmark_id, metadata in self.topological_graph.landmark_metadata.items():
                reference_path = self.topological_graph.get_landmark_reference_image(landmark_id)
                landmark_image_info[str(landmark_id)] = {
                    'has_reference_image': reference_path is not None,
                    'reference_image_path': str(reference_path) if reference_path else None,
                    'total_occurrences': metadata.get('total_occurrences', 0),
                    'average_quality': metadata.get('average_quality', 0),
                    'image_occurrences': len([occ for occ in metadata.get('occurrences', [])
                                              if occ.get('landmark_image_path')]),
                    'scene_indoor_outdoor': metadata.get('scene_indoor_outdoor', 'unknown'),
                    'scene_category': metadata.get('scene_category', 'unknown'),
                }

            storage_info = {
                'timestamp': datetime.now().isoformat(),
                'segmentation_model': 'FastSAM',
                'covisibility_method': 'geometric_distance_based',
                'scene_analysis_enabled': self.processing_stats['scene_analysis_enabled'],
                'storage_statistics': {
                    'total_landmarks': total_landmarks,
                    'landmarks_with_images': landmarks_with_images,
                    'total_landmark_images': total_images,
                    'image_coverage_rate': landmarks_with_images / max(total_landmarks, 1),
                    'extractor_session_dir': str(self.landmark_extractor.get_session_directory()),
                    'scene_analysis_success_rate': self.processing_stats.get('scene_analysis_success_rate', 0.0),
                },
                'landmark_image_details': landmark_image_info,
            }

            info_path = save_dir / "image_storage_info.json"
            with open(info_path, 'w') as f:
                json.dump(storage_info, f, indent=2, default=str)

        except Exception as e:
            print(f"Failed to save image storage info: {e}")

    def _save_landmark_visualizations(self, save_dir):
        """Save landmark visualizations for each image."""
        try:
            viz_dir = save_dir / "landmark_visualizations"
            viz_dir.mkdir(exist_ok=True)

            for image_path, image_data in self.visualization_data['image_landmarks'].items():
                landmarks = image_data['landmarks']
                image_index = image_data['image_index']
                scene_info = image_data.get('scene_info') or {}

                image = cv2.imread(image_path)
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                fig, ax = plt.subplots(1, 1, figsize=(16, 12))
                ax.imshow(image_rgb)

                if landmarks:
                    colors = plt.cm.Set3(np.linspace(0, 1, len(landmarks)))

                    for i, (landmark, color) in enumerate(zip(landmarks, colors)):
                        bbox = landmark['bbox']
                        center = landmark['center']
                        quality = landmark.get('quality_score', 0)
                        area = landmark.get('area', 0)
                        has_image = landmark.get('landmark_image_path') is not None
                        confidence = landmark.get('confidence', 0)

                        has_siglip = 'siglip' in landmark.get('features', {})
                        has_dinov2 = 'dinov2' in landmark.get('features', {})

                        rect = Rectangle(
                            (bbox[0], bbox[1]), bbox[2], bbox[3],
                            linewidth=3, edgecolor=color, facecolor='none', alpha=0.8
                        )
                        ax.add_patch(rect)

                        ax.plot(center[0], center[1], 'o', color=color, markersize=8,
                                markeredgecolor='white', markeredgewidth=2)

                        feature_info = ""
                        if has_siglip:
                            feature_info += "S"
                        if has_dinov2:
                            feature_info += "D"
                        if not feature_info:
                            feature_info = "X"

                        image_info = "Img" if has_image else "NoImg"
                        label = f"L{i + 1}({feature_info})\nQ:{quality:.2f}\nC:{confidence:.2f}\nA:{area:.0f}\n{image_info}"
                        ax.annotate(label, (bbox[0], bbox[1]),
                                    xytext=(5, 5), textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                                    fontsize=10, fontweight='bold')

                    landmarks_with_images = len([lm for lm in landmarks if lm.get('landmark_image_path')])
                else:
                    ax.text(0.5, 0.5, 'NO LANDMARKS DETECTED',
                            ha='center', va='center', transform=ax.transAxes,
                            bbox=dict(boxstyle='round,pad=1', facecolor='red', alpha=0.7),
                            fontsize=16, fontweight='bold', color='white')
                    landmarks_with_images = 0

                image_name = Path(image_path).name
                scene_text = ""
                if scene_info.get('analysis_success', False):
                    indoor_outdoor = scene_info.get('indoor_outdoor', 'unknown')
                    scene_category = scene_info.get('scene_category', 'unknown')
                    scene_text = f"Scene: {indoor_outdoor}/{scene_category}"

                title = (f"Image {image_index + 1}: {image_name}\n"
                         f"{len(landmarks)} landmarks (S=SigLIP, D=DINOv2, Img=Stored Image)\n"
                         f"FastSAM Segmentation | Image Storage: {landmarks_with_images}/{len(landmarks)}")

                if scene_text:
                    title += f"\n{scene_text}"

                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                ax.axis('off')

                output_path = viz_dir / f"image_{image_index + 1:02d}_{Path(image_path).stem}_landmarks.png"
                plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close()

        except Exception as e:
            print(f"Landmark visualization failed: {e}")

    def _save_merge_pair_visualizations(self, save_dir):
        """Save merge pair visualizations."""
        try:
            merge_dir = save_dir / "merge_pair_visualizations"
            merge_dir.mkdir(exist_ok=True)

            merge_pairs = self.visualization_data['merge_pairs']

            if not merge_pairs:
                return

            for i, merge_pair in enumerate(merge_pairs):
                try:
                    current_image = cv2.imread(merge_pair['current_image'])
                    if current_image is None:
                        continue
                    current_rgb = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

                    target_image = cv2.imread(merge_pair['target_image'])
                    if target_image is None:
                        continue
                    target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

                    current_landmarks = self.visualization_data['image_landmarks'][merge_pair['current_image']][
                        'landmarks']
                    current_landmark = current_landmarks[merge_pair['current_landmark_local_id']]

                    fig = plt.figure(figsize=(20, 12))

                    ax1 = plt.subplot2grid((2, 2), (0, 0))
                    ax2 = plt.subplot2grid((2, 2), (0, 1))

                    ax1.imshow(target_rgb)
                    target_bbox = merge_pair['target_bbox']
                    target_center = merge_pair['target_center']

                    target_rect = Rectangle(
                        (target_bbox[0], target_bbox[1]), target_bbox[2], target_bbox[3],
                        linewidth=4, edgecolor='lime', facecolor='none', alpha=0.9
                    )
                    ax1.add_patch(target_rect)
                    ax1.plot(target_center[0], target_center[1], 'o', color='lime', markersize=12,
                             markeredgecolor='white', markeredgewidth=3)

                    target_name = Path(merge_pair['target_image']).name
                    ax1.set_title(f"Original Landmark (L{merge_pair['target_landmark_id']})\n{target_name}",
                                  fontsize=12, fontweight='bold')
                    ax1.axis('off')

                    ax2.imshow(current_rgb)
                    current_bbox = current_landmark['bbox']
                    current_center = current_landmark['center']

                    current_rect = Rectangle(
                        (current_bbox[0], current_bbox[1]), current_bbox[2], current_bbox[3],
                        linewidth=4, edgecolor='orange', facecolor='none', alpha=0.9
                    )
                    ax2.add_patch(current_rect)
                    ax2.plot(current_center[0], current_center[1], 'o', color='orange', markersize=12,
                             markeredgecolor='white', markeredgewidth=3)

                    current_name = Path(merge_pair['current_image']).name
                    ax2.set_title(f"Matched Landmark\n{current_name}",
                                  fontsize=12, fontweight='bold')
                    ax2.axis('off')

                    similarity_score = merge_pair['similarity_score']
                    similarity_details = merge_pair.get('similarity_details', {})

                    components = self._get_similarity_components(similarity_details)
                    component_names = [comp[0] for comp in components]
                    component_scores = [comp[1] for comp in components]

                    bar_ax = fig.add_subplot(2, 1, 2)
                    colors = ['red', 'orange', 'green', 'blue']
                    bars = bar_ax.bar(component_names, component_scores,
                                      color=colors[:len(component_names)], alpha=0.7)

                    for bar, score in zip(bars, component_scores):
                        if score > 0:
                            bar_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

                    bar_ax.set_ylim(0, 1.0)
                    bar_ax.set_ylabel('Similarity Score', fontweight='bold')
                    bar_ax.set_title(f'SigLIP Feature Similarity Analysis (FastSAM + Scene Aware)\nOverall Score: {similarity_score:.3f}',
                                     fontsize=14, fontweight='bold')
                    bar_ax.grid(True, alpha=0.3)

                    fig.suptitle(f'SigLIP Feature Match #{i + 1} - L{merge_pair["target_landmark_id"]}',
                                 fontsize=16, fontweight='bold', y=0.98)

                    plt.tight_layout()

                    output_path = merge_dir / f"merge_{i + 1:02d}_L{merge_pair['target_landmark_id']}_sim{similarity_score:.3f}.png"
                    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
                    plt.close()

                except Exception:
                    continue

        except Exception as e:
            print(f"Merge pair visualization failed: {e}")

    def _get_similarity_components(self, similarity_details):
        """Get similarity components for visualization."""
        components = [
            ('SigLIP', similarity_details.get('siglip_similarity', 0.0)),
            ('DINOv2', similarity_details.get('dinov2_similarity', 0.0)),
            ('Spatial', similarity_details.get('spatial_consistency', 0.0)),
            ('Scale', similarity_details.get('scale_consistency', 0.0))
        ]
        return components

    def _save_graph_visualization(self, save_dir):
        """Save graph visualization."""
        try:
            viz_path = save_dir / "graph_visualization_geometric_covisibility_scene_aware.png"
            self.topological_graph.visualize_graph(str(viz_path))
        except Exception as e:
            print(f"Graph visualization failed: {e}")

    def _save_system_info(self, save_dir):
        """Save system configuration."""
        try:
            config_path = save_dir / "system_config.json"
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'feature_alignment': 'siglip_aligned',
                'segmentation_model': 'FastSAM',
                'image_storage_enabled': True,
                'multi_image_navigation_ready': True,
                'covisibility_method': 'geometric_distance_based',
                'scene_analysis_enabled': self.processing_stats['scene_analysis_enabled'],
                'scene_analysis_success_rate': self.processing_stats.get('scene_analysis_success_rate', 0.0),
                'processing_statistics': self.processing_stats,
                'graph_statistics': self.topological_graph.stats,
                'visualization_stats': {
                    'total_merge_pairs': len(self.visualization_data['merge_pairs']),
                    'images_with_landmarks': len(self.visualization_data['image_landmarks']),
                    'total_images_processed': len(self.visualization_data['image_paths']),
                },
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Config save failed: {e}")

    def _visualize_query_results(self, query: str, results: List, top_k: int = 3):
        """Visualize query results."""
        try:
            viz_dir = get_safe_output_dir() / "query_results"
            viz_dir.mkdir(exist_ok=True)

            fig, axes = plt.subplots(1, top_k, figsize=(6 * top_k, 8))
            if top_k == 1:
                axes = [axes]

            for i, (landmark_id, similarity_score) in enumerate(results[:top_k]):
                ax = axes[i]

                reference_image_path = self.topological_graph.get_landmark_reference_image(landmark_id)

                if reference_image_path and Path(reference_image_path).exists():
                    try:
                        landmark_image = cv2.imread(reference_image_path)
                        if landmark_image is not None:
                            landmark_rgb = cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB)
                            ax.imshow(landmark_rgb)

                            h, w = landmark_rgb.shape[:2]
                            rect = Rectangle((0, 0), w - 1, h - 1, linewidth=4, edgecolor='lime', facecolor='none',
                                             alpha=0.9)
                            ax.add_patch(rect)

                            metadata = self.topological_graph.landmark_metadata.get(landmark_id, {})
                            scene_indoor_outdoor = metadata.get('scene_indoor_outdoor', 'unknown')
                            scene_category = metadata.get('scene_category', 'unknown')

                            label_text = f"Sim: {similarity_score:.3f}\nStored Image"
                            if scene_indoor_outdoor != 'unknown':
                                label_text += f"\n{scene_indoor_outdoor}/{scene_category}"

                            ax.text(10, 30, label_text,
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                                    fontsize=10, fontweight='bold')

                            title = f'L{landmark_id} (#{i + 1})\nStored Reference\nSim: {similarity_score:.3f}'
                            if scene_indoor_outdoor != 'unknown':
                                title += f'\n{scene_indoor_outdoor}/{scene_category}'

                            ax.set_title(title, fontsize=12, fontweight='bold')
                        else:
                            ax.text(0.5, 0.5, f'Failed to load stored image', ha='center', va='center',
                                    transform=ax.transAxes)
                            ax.set_title(f'L{landmark_id}: {similarity_score:.3f}')
                    except Exception:
                        ax.text(0.5, 0.5, f'Error loading stored image', ha='center', va='center',
                                transform=ax.transAxes)
                        ax.set_title(f'L{landmark_id}: {similarity_score:.3f}')
                else:
                    landmark_metadata = self.topological_graph.landmark_metadata.get(landmark_id)
                    if not landmark_metadata or not landmark_metadata['occurrences']:
                        ax.text(0.5, 0.5, f'Landmark {landmark_id}\nNo image data',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'L{landmark_id}: {similarity_score:.3f}')
                        continue

                    best_occurrence = max(landmark_metadata['occurrences'],
                                          key=lambda x: x['quality_score'])

                    image_path = best_occurrence['image_path']
                    bbox = best_occurrence['bbox']
                    center = best_occurrence['center']
                    quality = best_occurrence['quality_score']

                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            ax.imshow(image_rgb)

                            rect = Rectangle(
                                (bbox[0], bbox[1]), bbox[2], bbox[3],
                                linewidth=4, edgecolor='lime', facecolor='none', alpha=0.9
                            )
                            ax.add_patch(rect)

                            ax.plot(center[0], center[1], 'o', color='red', markersize=12,
                                    markeredgecolor='white', markeredgewidth=3)

                            scene_indoor_outdoor = landmark_metadata.get('scene_indoor_outdoor', 'unknown')
                            scene_category = landmark_metadata.get('scene_category', 'unknown')

                            label_text = f"Sim: {similarity_score:.3f}\nQuality: {quality:.3f}"
                            if scene_indoor_outdoor != 'unknown':
                                label_text += f"\n{scene_indoor_outdoor}/{scene_category}"

                            ax.annotate(label_text, (bbox[0], bbox[1]),
                                        xytext=(5, 5), textcoords='offset points',
                                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                                        fontsize=10, fontweight='bold')

                            title = f'L{landmark_id} (#{i + 1})\nOriginal Context\nSim: {similarity_score:.3f}'
                            if scene_indoor_outdoor != 'unknown':
                                title += f'\n{scene_indoor_outdoor}/{scene_category}'

                            ax.set_title(title, fontsize=12, fontweight='bold')
                        else:
                            ax.text(0.5, 0.5, f'Image not found\n{Path(image_path).name}',
                                    ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'L{landmark_id}: {similarity_score:.3f}')

                    except Exception:
                        ax.text(0.5, 0.5, f'Error loading image',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'L{landmark_id}: {similarity_score:.3f}')

                ax.axis('off')

            fig.suptitle(f"Top {top_k} Matches for Query: '{query}'\nSigLIP Aligned + FastSAM + Scene Aware",
                         fontsize=16, fontweight='bold', y=0.95)

            plt.tight_layout()

            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_query = safe_query.replace(' ', '_')
            output_path = viz_dir / f"query_{safe_query}_top{top_k}_results.png"
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()

            result_info = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'segmentation_model': 'FastSAM',
                'image_storage_enabled': True,
                'covisibility_method': 'geometric_distance_based',
                'scene_analysis_enabled': self.processing_stats['scene_analysis_enabled'],
                'results': []
            }

            for i, (landmark_id, similarity_score) in enumerate(results[:top_k]):
                landmark_metadata = self.topological_graph.landmark_metadata.get(landmark_id, {})
                has_stored_image = self.topological_graph.get_landmark_reference_image(landmark_id) is not None
                result_info['results'].append({
                    'rank': i + 1,
                    'landmark_id': landmark_id,
                    'similarity_score': similarity_score,
                    'occurrences': landmark_metadata.get('total_occurrences', 0),
                    'average_quality': landmark_metadata.get('average_quality', 0),
                    'has_stored_reference_image': has_stored_image,
                    'stored_image_path': self.topological_graph.get_landmark_reference_image(landmark_id),
                    'scene_indoor_outdoor': landmark_metadata.get('scene_indoor_outdoor', 'unknown'),
                    'scene_category': landmark_metadata.get('scene_category', 'unknown'),
                })

            info_path = viz_dir / f"query_{safe_query}_info.json"
            with open(info_path, 'w') as f:
                json.dump(result_info, f, indent=2, default=str)

        except Exception as e:
            print(f"Query result visualization failed: {e}")


def test_navigation_system_siglip_aligned(dataset_path: str,
                                          fastsam_model_path: str = "FastSAM-s.pt",
                                          max_images: int = 100,
                                          sample_interval: int = 10,
                                          use_fastsam: bool = True,
                                          quality_threshold: float = 0.65,
                                          preferred_gpu_id: int = None,
                                          siglip_model_path: str = "google/siglip-so400m-patch14-384",
                                          dinov2_model: str = None):
    """Test navigation system using SigLIP-aligned features and FastSAM."""

    print(f"Testing Navigation System with FastSAM")
    print(f"Dataset: {dataset_path}")

    valid, image_paths = validate_dataset_path(dataset_path)
    if not valid:
        print(f"Invalid dataset: {dataset_path}")
        return None

    print(f"Found {len(image_paths)} images")

    nav_system = LandmarkNavigationSystem(
        fastsam_model_path=fastsam_model_path,
        siglip_model_path=siglip_model_path,
        dinov2_model=dinov2_model,
        use_fastsam=use_fastsam,
        base_similarity_threshold=0.78,
        preferred_gpu_id=preferred_gpu_id
    )

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


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/dianzi2"
    fastsam_model_path = sys.argv[2] if len(sys.argv) > 2 else "FastSAM-s.pt"

    nav_system = test_navigation_system_siglip_aligned(
        dataset_path=dataset_path,
        fastsam_model_path=fastsam_model_path,
        max_images=20,
        sample_interval=1,
        quality_threshold=0.65
    )

    if nav_system:
        print("Test completed successfully!")
    else:
        print("Test failed")
