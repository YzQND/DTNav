#!/usr/bin/env python3
"""
Topological graph module using SigLIP feature alignment
Supports landmark image storage and multi-image navigation
Features: scene-aware topological graph construction, hybrid relationships,
optimized scene information with multi-observation fusion
"""

import time
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import networkx as nx
import matplotlib.pyplot as plt

from utils import monitor_gpu_memory, get_best_gpu, set_gpu_device


class LandmarkTopologicalGraph:
    """Topological graph using SigLIP-aligned features with scene-aware navigation"""

    def __init__(self, base_similarity_threshold=0.75, adaptive_threshold=True,
                 siglip_encoder=None, preferred_gpu_id=None,
                 merge_temporal_window=10):
        self.graph = nx.Graph()
        self.landmark_features = {}
        self.landmark_metadata = {}
        self.image_landmarks = {}

        if preferred_gpu_id is None:
            gpu_id, _, _ = get_best_gpu()
        else:
            gpu_id = preferred_gpu_id

        self.device = set_gpu_device(gpu_id)
        self.siglip_encoder = siglip_encoder

        self.base_similarity_threshold = base_similarity_threshold
        self.adaptive_threshold = adaptive_threshold
        self.learned_thresholds = {
            'siglip': 0.70,
            'dinov2': 0.75,
            'spatial': 0.65,
            'scale': 0.60,
            'combined': base_similarity_threshold
        }

        self.global_landmark_id = 0

        self.temporal_window_frames = 6
        self.frame_duration = 1.0

        self.stats = {
            'total_landmarks': 0,
            'merged_landmarks': 0,
            'rejected_merges': 0,
            'covisibility_connections': 0,
            'temporal_connections': 0,
            'processed_images': 0,
            'language_queries': 0,
            'successful_language_matches': 0,
            'siglip_similarity_distribution': [],
            'dinov2_similarity_distribution': [],
            'spatial_similarity_distribution': [],
            'scale_similarity_distribution': [],
            'merge_rejection_reasons': defaultdict(int),
            'landmarks_with_images': 0,
            'total_landmark_images': 0,
            'geometric_distance_distribution': [],
            'temporal_distance_distribution': [],
            'scene_analysis_stats': {
                'indoor_landmarks': 0,
                'outdoor_landmarks': 0,
                'unknown_scene_landmarks': 0,
                'scene_categories': defaultdict(int),
                'boundary_landmarks': 0,
                'multi_scene_landmarks': 0,
            },
            'scene_query_stats': {
                'total_scene_queries': 0,
                'successful_scene_queries': 0,
                'scene_entry_selections': 0,
                'scene_types_queried': defaultdict(int),
                'boundary_landmark_selections': 0,
                'scene_match_score_distribution': [],
            }
        }

        self.semantic_clusters = defaultdict(list)
        self.cluster_representatives = {}

        self.scene_index_cache = {}
        self.scene_cache_timestamp = 0

        self.image_timestamps = {}
        self.sorted_image_paths = []

        self.merge_temporal_window = merge_temporal_window

        self.landmark_creation_info = {}
        self.current_frame_index = 0

    def _parse_landmark_id_from_name(self, landmark_name: str) -> Optional[int]:
        """Parse landmark ID from landmark name"""
        import re

        patterns = [
            r'landmark_(\d+)$',
            r'L(\d+)$',
        ]

        for pattern in patterns:
            match = re.search(pattern, landmark_name.strip())
            if match:
                landmark_id = int(match.group(1))
                if landmark_id in self.landmark_metadata:
                    return landmark_id

        return None

    def query_landmarks_by_language(self, instruction: str, top_k: int = 5):
        """Query landmarks using natural language"""
        if not self.landmark_features:
            return []

        self.stats['language_queries'] += 1

        try:
            direct_landmark_id = self._parse_landmark_id_from_name(instruction)
            if direct_landmark_id is not None:
                self.stats['successful_language_matches'] += 1
                return [(direct_landmark_id, 1.0)]

            if not self.siglip_encoder or not self.siglip_encoder.model:
                return []

            visual_features_list = []
            for landmark_id, landmark_features in self.landmark_features.items():
                if 'siglip' in landmark_features and landmark_features['siglip'] is not None:
                    visual_features_list.append((landmark_id, landmark_features['siglip']))

            if not visual_features_list:
                return []

            similarities = self.siglip_encoder.compute_similarity(
                instruction, visual_features_list
            )

            if not similarities:
                return []

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]

            if top_results:
                self.stats['successful_language_matches'] += 1

            return top_results

        except Exception as e:
            return []

    def get_landmarks_by_scene_type(self, scene_name: str) -> List[int]:
        """Query landmarks by scene type using probability-based matching"""
        try:
            self.stats['scene_query_stats']['total_scene_queries'] += 1
            self.stats['scene_query_stats']['scene_types_queried'][scene_name.lower()] += 1

            cache_key = scene_name.lower()
            current_time = time.time()

            if (cache_key in self.scene_index_cache and
                    current_time - self.scene_cache_timestamp < 60):
                cached_landmarks = self.scene_index_cache[cache_key]
                if cached_landmarks:
                    self.stats['scene_query_stats']['successful_scene_queries'] += 1
                return cached_landmarks

            scene_landmarks = []
            scene_name_variations = self._generate_scene_name_variations(scene_name)

            landmark_scores = []
            for landmark_id, metadata in self.landmark_metadata.items():
                score = self._calculate_scene_match_score(landmark_id, scene_name_variations)
                if score > 0.1:
                    landmark_scores.append((landmark_id, score))

            landmark_scores.sort(key=lambda x: x[1], reverse=True)
            scene_landmarks = [landmark_id for landmark_id, score in landmark_scores]

            scores = [score for _, score in landmark_scores]
            self.stats['scene_query_stats']['scene_match_score_distribution'].extend(scores)

            self.scene_index_cache[cache_key] = scene_landmarks
            self.scene_cache_timestamp = current_time

            if scene_landmarks:
                self.stats['scene_query_stats']['successful_scene_queries'] += 1

            return scene_landmarks

        except Exception:
            return []

    def _calculate_scene_match_score(self, landmark_id: int, scene_name_variations: List[str]) -> float:
        """Calculate scene matching score based on scene probability"""
        try:
            metadata = self.landmark_metadata.get(landmark_id)
            if not metadata:
                return 0.0

            scene_observations = metadata.get('scene_observations', {})
            if not scene_observations:
                return self._legacy_scene_match_score(metadata, scene_name_variations)

            total_observations = sum(obs['count'] for obs in scene_observations.values())
            if total_observations == 0:
                return 0.0

            max_score = 0.0

            for variation in scene_name_variations:
                for observed_scene, obs_data in scene_observations.items():
                    if self._scene_names_match(variation, observed_scene):
                        scene_prob = obs_data['count'] / total_observations
                        avg_confidence = obs_data['total_confidence'] / obs_data['count']
                        quality_factor = metadata.get('average_quality', 0.5)
                        boundary_bonus = 1.1 if metadata.get('is_boundary_landmark', False) else 1.0
                        score = scene_prob * avg_confidence * quality_factor * boundary_bonus
                        max_score = max(max_score, score)

            return max_score

        except Exception:
            return 0.0

    def _legacy_scene_match_score(self, metadata: dict, scene_name_variations: List[str]) -> float:
        """Legacy scene matching for backward compatibility"""
        scene_indoor_outdoor = metadata.get('scene_indoor_outdoor', 'unknown').lower()
        scene_category = metadata.get('scene_category', 'unknown').lower()
        scene_confidence = metadata.get('scene_confidence', 0.0)

        for variation in scene_name_variations:
            if (variation in scene_category or variation in scene_indoor_outdoor or
                    scene_category in variation or scene_indoor_outdoor in variation):
                quality_factor = metadata.get('average_quality', 0.5)
                return scene_confidence * quality_factor

        return 0.0

    def _scene_names_match(self, variation: str, observed_scene: str) -> bool:
        """Check if scene names match"""
        variation = variation.lower()
        observed_scene = observed_scene.lower()

        if variation == observed_scene:
            return True

        if len(variation) > 3 and len(observed_scene) > 3:
            if variation in observed_scene or observed_scene in variation:
                return True

        return False

    def _generate_scene_name_variations(self, scene_name: str) -> List[str]:
        """Generate scene name variations for matching"""
        variations = [scene_name.lower().strip()]

        scene_mappings = {
            'kitchen': ['cook', 'dining'],
            'bedroom': ['sleep', 'bed'],
            'living room': ['lounge', 'sitting'],
            'bathroom': ['toilet', 'wash'],
            'office': ['work', 'study'],
            'garden': ['yard', 'outdoor'],
            'corridor': ['hallway', 'passage'],
            'garage': ['parking'],
            'balcony': ['terrace'],
            'basement': ['cellar'],
            'indoor': ['inside', 'interior'],
            'outdoor': ['outside', 'exterior'],
        }

        for key, values in scene_mappings.items():
            if scene_name.lower() in key or key in scene_name.lower():
                variations.extend(values)
            for value in values:
                if scene_name.lower() in value or value in scene_name.lower():
                    variations.extend([key] + [v for v in values if v != value])

        variations = list(set([v for v in variations if v and len(v) > 1]))

        return variations

    def find_closest_scene_entry_to_current(self, scene_landmarks: List[int]) -> Optional[int]:
        """Find closest scene entry to current observations"""
        try:
            if not scene_landmarks:
                return None

            self.stats['scene_query_stats']['scene_entry_selections'] += 1

            current_landmarks = []
            if self.image_landmarks:
                latest_image = max(self.image_landmarks.keys(),
                                   key=lambda x: self.image_landmarks[x])
                current_landmarks = self.image_landmarks.get(latest_image, [])

            if not current_landmarks:
                return self._select_highest_quality_landmark(scene_landmarks)

            best_entry = None
            min_avg_distance = float('inf')

            for scene_landmark_id in scene_landmarks:
                if scene_landmark_id not in self.landmark_metadata:
                    continue

                distances = []
                for current_landmark_id in current_landmarks:
                    if current_landmark_id in self.landmark_metadata:
                        distance = self._compute_landmark_distance(
                            scene_landmark_id, current_landmark_id
                        )
                        if distance < float('inf'):
                            distances.append(distance)

                if distances:
                    avg_distance = np.mean(distances)
                    if avg_distance < min_avg_distance:
                        min_avg_distance = avg_distance
                        best_entry = scene_landmark_id

            if best_entry and self.landmark_metadata.get(best_entry, {}).get('is_boundary_landmark', False):
                self.stats['scene_query_stats']['boundary_landmark_selections'] += 1

            return best_entry if best_entry is not None else self._select_highest_quality_landmark(scene_landmarks)

        except Exception:
            return self._select_highest_quality_landmark(scene_landmarks) if scene_landmarks else None

    def find_closest_scene_entry_to_landmark(self, scene_landmarks: List[int],
                                             reference_landmark_id: int) -> Optional[int]:
        """Find closest scene entry to a specified landmark"""
        try:
            if not scene_landmarks or reference_landmark_id not in self.landmark_metadata:
                return None

            self.stats['scene_query_stats']['scene_entry_selections'] += 1

            best_entry = None
            min_distance = float('inf')

            for scene_landmark_id in scene_landmarks:
                if scene_landmark_id not in self.landmark_metadata:
                    continue

                distance = self._compute_landmark_distance(scene_landmark_id, reference_landmark_id)

                if distance < min_distance:
                    min_distance = distance
                    best_entry = scene_landmark_id

            if best_entry and self.landmark_metadata.get(best_entry, {}).get('is_boundary_landmark', False):
                self.stats['scene_query_stats']['boundary_landmark_selections'] += 1

            return best_entry if best_entry is not None else self._select_highest_quality_landmark(scene_landmarks)

        except Exception:
            return self._select_highest_quality_landmark(scene_landmarks) if scene_landmarks else None

    def _compute_landmark_distance(self, landmark1_id: int, landmark2_id: int) -> float:
        """Compute distance between two landmarks"""
        try:
            if self.graph.has_edge(landmark1_id, landmark2_id):
                edge_data = self.graph.get_edge_data(landmark1_id, landmark2_id)
                return edge_data.get('geometric_distance', 1.0)

            try:
                distance = nx.shortest_path_length(
                    self.graph, landmark1_id, landmark2_id, weight='geometric_distance'
                )
                return distance
            except nx.NetworkXNoPath:
                return self._estimate_spatial_distance(landmark1_id, landmark2_id)

        except Exception:
            return float('inf')

    def _estimate_spatial_distance(self, landmark1_id: int, landmark2_id: int) -> float:
        """Estimate spatial distance based on landmark centers"""
        try:
            metadata1 = self.landmark_metadata.get(landmark1_id)
            metadata2 = self.landmark_metadata.get(landmark2_id)

            if not metadata1 or not metadata2:
                return float('inf')

            center1 = metadata1.get('representative_center', [0, 0])
            center2 = metadata2.get('representative_center', [0, 0])

            if len(center1) >= 2 and len(center2) >= 2:
                dx = center1[0] - center2[0]
                dy = center1[1] - center2[1]
                distance = np.sqrt(dx * dx + dy * dy)
                return min(2.0, distance / 500.0)

            return 1.0

        except Exception:
            return float('inf')

    def _select_highest_quality_landmark(self, landmarks: List[int]) -> Optional[int]:
        """Select highest quality landmark from list"""
        try:
            if not landmarks:
                return None

            best_landmark = None
            best_quality = 0.0

            for landmark_id in landmarks:
                metadata = self.landmark_metadata.get(landmark_id)
                if metadata:
                    quality = metadata.get('average_quality', 0.0)
                    occurrences = metadata.get('total_occurrences', 0)
                    combined_score = quality * min(1.5, 1.0 + occurrences * 0.1)

                    if combined_score > best_quality:
                        best_quality = combined_score
                        best_landmark = landmark_id

            return best_landmark

        except Exception:
            return landmarks[0] if landmarks else None

    def get_landmark_representative_name(self, landmark_id: int) -> Optional[str]:
        """Get representative name for landmark"""
        try:
            metadata = self.landmark_metadata.get(landmark_id)
            if not metadata:
                return None

            dominant_scene = metadata.get('dominant_scene', 'unknown')
            if dominant_scene != 'unknown' and dominant_scene != '':
                return f"{dominant_scene}_landmark_{landmark_id}"

            scene_category = metadata.get('scene_category', 'unknown')
            scene_indoor_outdoor = metadata.get('scene_indoor_outdoor', 'unknown')

            if scene_category != 'unknown' and scene_category != '':
                return f"{scene_category}_landmark_{landmark_id}"
            elif scene_indoor_outdoor != 'unknown':
                return f"{scene_indoor_outdoor}_landmark_{landmark_id}"

            quality = metadata.get('average_quality', 0.0)
            area = metadata.get('average_area', 0.0)

            if quality > 0.8:
                return f"high_quality_landmark_{landmark_id}"
            elif area > 5000:
                return f"large_landmark_{landmark_id}"
            else:
                return f"landmark_{landmark_id}"

        except Exception:
            return f"landmark_{landmark_id}"

    def _update_derived_scene_info(self, landmark_id: int):
        """Update derived scene information (entropy, dominant scene, boundary detection)"""
        try:
            metadata = self.landmark_metadata.get(landmark_id)
            if not metadata:
                return

            scene_observations = metadata.get('scene_observations', {})
            if not scene_observations:
                return

            total_count = sum(obs['count'] for obs in scene_observations.values())
            if total_count == 0:
                return

            entropy = 0.0
            dominant_scene = None
            max_prob = 0.0

            for scene, obs in scene_observations.items():
                prob = obs['count'] / total_count
                if prob > 0:
                    entropy -= prob * np.log2(prob)

                if prob > max_prob:
                    max_prob = prob
                    dominant_scene = scene

            metadata['scene_entropy'] = entropy
            metadata['dominant_scene'] = dominant_scene if dominant_scene else 'unknown'

            is_boundary = entropy > 1.0 and max_prob < 0.7
            was_boundary = metadata.get('is_boundary_landmark', False)
            metadata['is_boundary_landmark'] = is_boundary

            unique_scenes = len([scene for scene, obs in scene_observations.items()
                                 if obs['count'] > 0])
            is_multi_scene = unique_scenes > 1
            metadata['is_multi_scene_landmark'] = is_multi_scene

            if is_boundary and not was_boundary:
                self.stats['scene_analysis_stats']['boundary_landmarks'] += 1
            elif not is_boundary and was_boundary:
                self.stats['scene_analysis_stats']['boundary_landmarks'] -= 1

            if is_multi_scene:
                self.stats['scene_analysis_stats']['multi_scene_landmarks'] += 1

        except Exception:
            pass

    def get_landmark_path_to_target(self, current_image_path: str, target_instruction: str):
        """Get navigation path to target"""
        target_landmarks = self.query_landmarks_by_language(target_instruction, top_k=3)
        if not target_landmarks:
            return None

        target_landmark_id = target_landmarks[0][0]

        current_landmarks = self.image_landmarks.get(current_image_path, [])
        if not current_landmarks:
            return None

        shortest_path = None
        shortest_distance = float('inf')

        for current_landmark_id in current_landmarks:
            if current_landmark_id == target_landmark_id:
                return [current_landmark_id]

            try:
                path = nx.shortest_path(self.graph, current_landmark_id, target_landmark_id,
                                        weight='geometric_distance')
                path_distance = nx.shortest_path_length(self.graph, current_landmark_id, target_landmark_id,
                                                        weight='geometric_distance')

                if path_distance < shortest_distance:
                    shortest_distance = path_distance
                    shortest_path = path

            except nx.NetworkXNoPath:
                continue

        return shortest_path

    def match_image_landmarks_without_adding(self, landmarks: List[Dict]) -> List[Tuple[int, float]]:
        """Match image landmarks without adding to graph"""
        if not landmarks:
            return []

        matched_landmarks = []

        for landmark in landmarks:
            similar_landmark_id, similarity_score, similarity_details = self._find_best_match(
                landmark, None
            )

            if similar_landmark_id is not None:
                validation_result, _ = self._validate_merge(
                    landmark, self.landmark_metadata[similar_landmark_id],
                    similarity_score, similarity_details
                )

                if validation_result:
                    matched_landmarks.append((similar_landmark_id, similarity_score))

        return matched_landmarks

    def find_shortest_path_from_current_to_target(self, current_matched_landmarks: List[Tuple[int, float]],
                                                  target_landmark_id: int) -> Tuple[Optional[int], Optional[List[int]]]:
        """Find shortest path based on geometric distance"""
        if not current_matched_landmarks:
            return None, None

        best_path = None
        best_score = float('inf')
        best_start_landmark = None

        for landmark_id, similarity_score in current_matched_landmarks:
            if landmark_id == target_landmark_id:
                return landmark_id, [landmark_id]

            try:
                path = nx.shortest_path(self.graph, landmark_id, target_landmark_id, weight='geometric_distance')
                geometric_distance = nx.shortest_path_length(self.graph, landmark_id, target_landmark_id,
                                                             weight='geometric_distance')
                final_score = geometric_distance

                if final_score < best_score:
                    best_score = final_score
                    best_path = path
                    best_start_landmark = landmark_id

            except nx.NetworkXNoPath:
                continue

        return best_start_landmark, best_path

    def get_landmark_reference_image(self, landmark_id: int) -> Optional[str]:
        """Get reference image path for landmark"""
        metadata = self.landmark_metadata.get(landmark_id)
        if not metadata:
            return None

        project_root = "/root/autodl-tmp/project"

        if 'representative_image_path' in metadata and metadata['representative_image_path']:
            reference_path = metadata['representative_image_path']

            if not Path(reference_path).is_absolute():
                full_path = Path(project_root) / reference_path
            else:
                full_path = Path(reference_path)

            if full_path.exists():
                return str(full_path)

        if 'occurrences' in metadata and metadata['occurrences']:
            best_occurrence = max(metadata['occurrences'], key=lambda x: x['quality_score'])
            landmark_image_path = best_occurrence.get('landmark_image_path')

            if landmark_image_path:
                if not Path(landmark_image_path).is_absolute():
                    full_path = Path(project_root) / landmark_image_path
                else:
                    full_path = Path(landmark_image_path)

                if full_path.exists():
                    return str(full_path)

        return None

    def get_current_view_landmarks_with_images(self, current_image_path: str) -> List[Tuple[int, str]]:
        """Get landmarks with reference images in current view"""
        current_landmarks = self.image_landmarks.get(current_image_path, [])
        landmarks_with_images = []

        for landmark_id in current_landmarks:
            reference_path = self.get_landmark_reference_image(landmark_id)
            if reference_path:
                landmarks_with_images.append((landmark_id, reference_path))

        return landmarks_with_images

    def add_image_landmarks(self, image_path: str, landmarks: List[Dict], timestamp: float = None,
                            scene_info: dict = None):
        """Add image landmarks to graph"""
        if timestamp is None:
            timestamp = time.time()

        if not landmarks:
            return {'attempted_merges': 0, 'successful_merges': 0, 'merge_details': []}

        self.image_timestamps[image_path] = timestamp
        self.sorted_image_paths = sorted(self.image_timestamps.keys(), key=lambda x: self.image_timestamps[x])

        self.current_frame_index = len(self.sorted_image_paths) - 1

        if scene_info is None:
            scene_info = {
                'indoor_outdoor': 'unknown',
                'scene_category': 'unknown',
                'confidence': 0.0,
                'analysis_success': False
            }

        image_landmark_ids = []
        merge_details = []
        attempted_merges = 0
        successful_merges = 0

        for i, landmark in enumerate(landmarks):
            similar_landmark_id, similarity_score, similarity_details = self._find_best_match(
                landmark, image_path
            )

            merge_detail = {
                'local_id': i,
                'landmark_data': landmark,
                'similarity_score': similarity_score,
                'similarity_details': similarity_details,
                'action': None,
                'target_id': None,
                'rejection_reason': None,
                'scene_info': scene_info,
            }

            if similar_landmark_id is not None:
                attempted_merges += 1

                merge_success, rejection_reason = self._intelligent_merge(
                    similar_landmark_id, landmark, image_path, timestamp,
                    similarity_score, similarity_details, scene_info=scene_info
                )

                if merge_success:
                    image_landmark_ids.append(similar_landmark_id)
                    successful_merges += 1
                    self.stats['merged_landmarks'] += 1
                    merge_detail['action'] = 'merged'
                    merge_detail['target_id'] = similar_landmark_id
                else:
                    new_id = self._create_landmark(landmark, image_path, timestamp, scene_info=scene_info)
                    image_landmark_ids.append(new_id)
                    self.stats['rejected_merges'] += 1
                    self.stats['merge_rejection_reasons'][rejection_reason] += 1
                    merge_detail['action'] = 'new'
                    merge_detail['target_id'] = new_id
                    merge_detail['rejection_reason'] = rejection_reason
            else:
                new_id = self._create_landmark(landmark, image_path, timestamp, scene_info=scene_info)
                image_landmark_ids.append(new_id)
                self.stats['total_landmarks'] += 1
                merge_detail['action'] = 'new'
                merge_detail['target_id'] = new_id
                merge_detail['rejection_reason'] = 'no_similar_landmark'

            merge_details.append(merge_detail)

        self._update_semantic_clusters(image_landmark_ids)

        self.image_landmarks[image_path] = image_landmark_ids
        self.stats['processed_images'] += 1

        self._add_hybrid_relationships(image_path, image_landmark_ids, timestamp)

        self.scene_index_cache.clear()
        self.scene_cache_timestamp = 0

        return {
            'attempted_merges': attempted_merges,
            'successful_merges': successful_merges,
            'merge_details': merge_details
        }

    def _calculate_frame_difference(self, image_path1: str, image_path2: str) -> int:
        """Calculate frame difference between two images"""
        try:
            if image_path1 not in self.image_timestamps or image_path2 not in self.image_timestamps:
                return float('inf')

            timestamp1 = self.image_timestamps[image_path1]
            timestamp2 = self.image_timestamps[image_path2]

            time_diff = abs(timestamp1 - timestamp2)
            frame_diff = int(time_diff / self.frame_duration)

            return frame_diff

        except Exception:
            return float('inf')

    def _add_hybrid_relationships(self, current_image_path: str, current_landmark_ids: List[int], timestamp: float):
        """Add hybrid relationships: covisibility + temporal"""
        self._add_covisibility_relationships(current_image_path, current_landmark_ids)
        self._add_temporal_relationships(current_image_path, current_landmark_ids)

    def _add_covisibility_relationships(self, image_path: str, landmark_ids: List[int]):
        """Add covisibility relationships (frame_diff = 0)"""
        for i, landmark1_id in enumerate(landmark_ids):
            for j, landmark2_id in enumerate(landmark_ids[i + 1:], i + 1):
                if not self.graph.has_edge(landmark1_id, landmark2_id):
                    landmark1_metadata = self.landmark_metadata[landmark1_id]
                    landmark2_metadata = self.landmark_metadata[landmark2_id]

                    geometric_distance = self._compute_geometric_distance(
                        landmark1_metadata, landmark2_metadata, image_path
                    )

                    normalized_distance = self._normalize_geometric_distance(geometric_distance)

                    self.graph.add_edge(
                        landmark1_id, landmark2_id,
                        type='covisibility',
                        geometric_distance=normalized_distance,
                        strength=1.0 - normalized_distance,
                        source_image=image_path,
                        raw_pixel_distance=geometric_distance,
                        frame_difference=0
                    )

                    self.stats['covisibility_connections'] += 1
                    self.stats['geometric_distance_distribution'].append(normalized_distance)

    def _add_temporal_relationships(self, current_image_path: str, current_landmark_ids: List[int]):
        """Add temporal relationships (frame_diff 1-5)"""
        for current_landmark_id in current_landmark_ids:
            current_metadata = self.landmark_metadata[current_landmark_id]

            for other_landmark_id, other_metadata in self.landmark_metadata.items():
                if other_landmark_id == current_landmark_id:
                    continue

                if self.graph.has_edge(current_landmark_id, other_landmark_id):
                    continue

                min_frame_diff = float('inf')
                best_temporal_distance = float('inf')

                for current_occ in current_metadata.get('occurrences', []):
                    current_img = current_occ['image_path']

                    for other_occ in other_metadata.get('occurrences', []):
                        other_img = other_occ['image_path']

                        if current_img == other_img:
                            continue

                        frame_diff = self._calculate_frame_difference(current_img, other_img)

                        if 1 <= frame_diff <= self.temporal_window_frames:
                            if frame_diff < min_frame_diff:
                                min_frame_diff = frame_diff
                                best_temporal_distance = self._compute_temporal_distance(
                                    frame_diff, current_occ, other_occ
                                )

                if min_frame_diff <= self.temporal_window_frames and best_temporal_distance < float('inf'):
                    self.graph.add_edge(
                        current_landmark_id, other_landmark_id,
                        type='temporal',
                        geometric_distance=best_temporal_distance,
                        strength=1.0 - best_temporal_distance,
                        frame_difference=min_frame_diff,
                        temporal_weight=self._compute_temporal_weight(min_frame_diff)
                    )

                    self.stats['temporal_connections'] += 1
                    self.stats['temporal_distance_distribution'].append(best_temporal_distance)

    def _compute_temporal_distance(self, frame_diff: int, occ1: dict, occ2: dict) -> float:
        """Compute temporal relationship distance weight"""
        try:
            temporal_decay = self._compute_temporal_weight(frame_diff)
            default_geometric_distance = 2.0

            temporal_distance = default_geometric_distance * (1 - temporal_decay) + (
                    frame_diff / self.temporal_window_frames)

            temporal_distance = max(1.0, temporal_distance)

            return temporal_distance

        except Exception:
            return 2.0

    def _compute_temporal_weight(self, frame_diff: int) -> float:
        """Compute temporal weight (smaller frame_diff = higher weight)"""
        if frame_diff <= 0:
            return 1.0
        elif frame_diff > self.temporal_window_frames:
            return 0.0
        else:
            return 1.0 - (frame_diff / self.temporal_window_frames)

    def _compute_geometric_distance(self, landmark1_metadata: Dict, landmark2_metadata: Dict,
                                    image_path: str) -> float:
        """Compute geometric distance between two landmarks in covisibility image"""
        center1 = None
        center2 = None

        for occurrence in landmark1_metadata.get('occurrences', []):
            if occurrence.get('image_path') == image_path:
                center1 = occurrence.get('center')
                break

        for occurrence in landmark2_metadata.get('occurrences', []):
            if occurrence.get('image_path') == image_path:
                center2 = occurrence.get('center')
                break

        if center1 is None or center2 is None:
            return float('inf')

        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        distance = np.sqrt(dx * dx + dy * dy)

        return distance

    def _normalize_geometric_distance(self, distance: float, image_shape: Tuple[int, int] = (1024, 1024)) -> float:
        """Normalize geometric distance"""
        if distance == float('inf'):
            return 1.0

        diagonal_length = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        normalized_distance = min(1.0, distance / diagonal_length)

        return normalized_distance

    def _find_best_match(self, landmark, current_image_path):
        """Find best match using SigLIP features"""
        if not self.landmark_features:
            return None, 0.0, {}

        best_match_id = None
        best_score = 0.0
        best_details = {}

        landmark_features = landmark['features']
        landmark_quality = landmark['quality_score']

        candidate_landmarks = self._get_recent_landmarks_for_merge()

        for landmark_id in candidate_landmarks:
            existing_features = self.landmark_features[landmark_id]
            existing_metadata = self.landmark_metadata[landmark_id]

            if current_image_path and any(
                    occ['image_path'] == current_image_path for occ in existing_metadata['occurrences']):
                continue

            similarity_score, similarity_details = self._compute_detailed_similarity(
                landmark_features, existing_features, landmark, existing_metadata
            )

            existing_quality = existing_metadata.get('average_quality', 0.5)
            quality_factor = min(1.2, 0.8 + 0.4 * min(landmark_quality, existing_quality))
            weighted_score = similarity_score * quality_factor

            if weighted_score > best_score:
                best_score = weighted_score
                best_match_id = landmark_id
                best_details = similarity_details
                best_details['quality_factor'] = quality_factor

        threshold = self.base_similarity_threshold
        if best_score > threshold:
            return best_match_id, best_score, best_details

        return None, best_score, best_details

    def _get_recent_landmarks_for_merge(self) -> List[int]:
        """Get landmarks within temporal window for merge comparison"""
        if self.merge_temporal_window <= 0:
            return list(self.landmark_features.keys())

        recent_landmarks = []
        current_frame = self.current_frame_index

        for landmark_id, creation_info in self.landmark_creation_info.items():
            creation_frame = creation_info['frame_index']
            frame_distance = current_frame - creation_frame

            if 0 <= frame_distance <= self.merge_temporal_window:
                recent_landmarks.append(landmark_id)

        return recent_landmarks

    def _get_dynamic_threshold(self, landmark_quality, similarity_details):
        """Get dynamic threshold"""
        base_threshold = self.learned_thresholds['combined']
        quality_adjustment = 0.05 * (landmark_quality - 0.5) if landmark_quality > 0.7 else 0
        semantic_bonus = 0.0
        if 'siglip_similarity' in similarity_details and similarity_details['siglip_similarity'] > 0.75:
            semantic_bonus = -0.05

        dynamic_threshold = base_threshold + quality_adjustment + semantic_bonus
        return max(0.55, min(0.85, dynamic_threshold))

    def _compute_detailed_similarity(self, features1, features2, landmark1_data, landmark2_metadata):
        """Compute detailed multi-modal similarity"""
        similarities = {}
        details = {}

        if 'siglip' in features1 and 'siglip' in features2:
            try:
                if features1['siglip'] is not None and features2['siglip'] is not None:
                    feat1 = features1['siglip'].to(self.device)
                    feat2 = features2['siglip'].to(self.device)

                    if feat1.dim() > 1 and feat1.shape[0] > 1:
                        feat1 = feat1.mean(dim=0, keepdim=True)
                    if feat2.dim() > 1 and feat2.shape[0] > 1:
                        feat2 = feat2.mean(dim=0, keepdim=True)

                    if feat1.dim() == 1:
                        feat1 = feat1.unsqueeze(0)
                    if feat2.dim() == 1:
                        feat2 = feat2.unsqueeze(0)

                    feat1_norm = F.normalize(feat1, p=2, dim=1)
                    feat2_norm = F.normalize(feat2, p=2, dim=1)
                    siglip_sim = torch.cosine_similarity(feat1_norm, feat2_norm, dim=1)

                    if siglip_sim.numel() > 1:
                        siglip_sim = siglip_sim.mean()

                    siglip_sim = siglip_sim.item()
                    siglip_sim = max(0.0, min(1.0, siglip_sim))

                    similarities['siglip'] = siglip_sim
                    details['siglip_similarity'] = siglip_sim
                    details['siglip_computation_method'] = 'cosine_similarity_for_image_matching'
                    self.stats['siglip_similarity_distribution'].append(siglip_sim)

            except Exception:
                pass

        if 'dinov2' in features1 and 'dinov2' in features2:
            try:
                if features1['dinov2'] is not None and features2['dinov2'] is not None:
                    feat1 = features1['dinov2'].to(self.device)
                    feat2 = features2['dinov2'].to(self.device)

                    if feat1.dim() > 1 and feat1.shape[0] > 1:
                        feat1 = feat1.mean(dim=0, keepdim=True)
                    if feat2.dim() > 1 and feat2.shape[0] > 1:
                        feat2 = feat2.mean(dim=0, keepdim=True)

                    if feat1.dim() == 1:
                        feat1 = feat1.unsqueeze(0)
                    if feat2.dim() == 1:
                        feat2 = feat2.unsqueeze(0)

                    dinov2_sim = torch.cosine_similarity(feat1, feat2, dim=1)
                    if dinov2_sim.numel() > 1:
                        dinov2_sim = dinov2_sim.mean()

                    dinov2_sim = max(0.0, min(1.0, dinov2_sim.item()))
                    similarities['dinov2'] = dinov2_sim
                    details['dinov2_similarity'] = dinov2_sim
                    self.stats['dinov2_similarity_distribution'].append(dinov2_sim)
            except:
                pass

        spatial_consistency = self._compute_spatial_consistency(landmark1_data, landmark2_metadata)
        similarities['spatial'] = spatial_consistency
        details['spatial_consistency'] = spatial_consistency
        self.stats['spatial_similarity_distribution'].append(spatial_consistency)

        scale_consistency = self._compute_scale_consistency(landmark1_data, landmark2_metadata)
        similarities['scale'] = scale_consistency
        details['scale_consistency'] = scale_consistency
        self.stats['scale_similarity_distribution'].append(scale_consistency)

        weights = self._get_adaptive_weights(similarities)
        details['weights_used'] = weights

        combined_score = 0.0
        total_weight = 0.0

        for modality, sim in similarities.items():
            if modality in weights:
                combined_score += weights[modality] * sim
                total_weight += weights[modality]

        final_score = combined_score / max(total_weight, 1.0)
        details['final_score'] = final_score
        details['component_count'] = len(similarities)

        return final_score, details

    def _get_adaptive_weights(self, similarities):
        """Get adaptive weights prioritizing SigLIP features"""
        weights = {}
        available_modalities = list(similarities.keys())

        base_weights = {'siglip': 0.20, 'dinov2': 0.70, 'spatial': 0.07, 'scale': 0.03}

        if len(available_modalities) == 4:
            return {mod: base_weights[mod] for mod in available_modalities if mod in base_weights}

        if 'siglip' in available_modalities and 'dinov2' in available_modalities:
            weights['siglip'] = 0.85
            weights['dinov2'] = 0.10
            remaining_weight = 0.05
            spatial_scale_modalities = [m for m in ['spatial', 'scale'] if m in available_modalities]
            if spatial_scale_modalities:
                weight_per_spatial = remaining_weight / len(spatial_scale_modalities)
                for modality in spatial_scale_modalities:
                    weights[modality] = weight_per_spatial
        elif 'siglip' in available_modalities:
            weights['siglip'] = 0.90
            remaining_weight = 0.10
            other_modalities = [m for m in available_modalities if m != 'siglip']
            if other_modalities:
                weight_per_other = remaining_weight / len(other_modalities)
                for modality in other_modalities:
                    weights[modality] = weight_per_other
        else:
            equal_weight = 1.0 / len(available_modalities)
            for modality in available_modalities:
                weights[modality] = equal_weight

        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _compute_spatial_consistency(self, landmark_data, existing_metadata):
        """Compute spatial consistency"""
        new_area = landmark_data['area']
        new_center = landmark_data['center']

        existing_areas = [occ['area'] for occ in existing_metadata['occurrences']]
        existing_centers = [occ['center'] for occ in existing_metadata['occurrences']]

        if not existing_areas:
            return 0.5

        avg_area = np.mean(existing_areas)
        area_ratio = min(new_area, avg_area) / max(new_area, avg_area)

        if existing_centers:
            distances = []
            for existing_center in existing_centers:
                dx = new_center[0] - existing_center[0]
                dy = new_center[1] - existing_center[1]
                distance = np.sqrt(dx * dx + dy * dy)
                distances.append(distance)

            avg_distance = np.mean(distances)
            typical_size = np.sqrt(avg_area)
            normalization_factor = max(typical_size * 3, 200)
            normalized_distance = min(1.0, avg_distance / normalization_factor)
            position_consistency = 1.0 - normalized_distance
        else:
            position_consistency = 0.5

        spatial_score = 0.7 * area_ratio + 0.3 * position_consistency
        return max(0.0, min(1.0, spatial_score))

    def _compute_scale_consistency(self, landmark_data, existing_metadata):
        """Compute scale consistency"""
        bbox1 = landmark_data['bbox']
        w1, h1 = bbox1[2], bbox1[3]

        if w1 <= 0 or h1 <= 0:
            return 0.0

        aspect1 = max(w1, h1) / min(w1, h1)

        existing_aspects = []
        for occ in existing_metadata['occurrences']:
            bbox = occ['bbox']
            w, h = bbox[2], bbox[3]
            if w > 0 and h > 0:
                aspect = max(w, h) / min(w, h)
                existing_aspects.append(aspect)

        if not existing_aspects:
            return 0.6

        avg_aspect = np.mean(existing_aspects)
        aspect_ratio = min(aspect1, avg_aspect) / max(aspect1, avg_aspect)

        return max(0.0, min(1.0, aspect_ratio))

    def _intelligent_merge(self, landmark_id, new_landmark, image_path, timestamp,
                           similarity_score, similarity_details, scene_info=None):
        """Intelligent merge decision"""
        existing_metadata = self.landmark_metadata[landmark_id]
        new_quality = new_landmark['quality_score']
        existing_quality = existing_metadata.get('average_quality', 0.5)

        validation_result, rejection_reason = self._validate_merge(
            new_landmark, existing_metadata, similarity_score, similarity_details
        )

        if not validation_result:
            return False, rejection_reason

        self._update_features_by_quality_selection(
            landmark_id, new_landmark, new_quality, existing_quality
        )

        self._update_metadata(landmark_id, new_landmark, image_path, timestamp, scene_info=scene_info)

        return True, "success"

    def _validate_merge(self, new_landmark, existing_metadata, similarity_score, similarity_details):
        """Validate merge decision"""
        threshold = self.base_similarity_threshold
        if similarity_score < threshold:
            return False, f"low_similarity({similarity_score:.3f}<{threshold:.3f})"

        new_area = new_landmark['area']
        existing_areas = [occ['area'] for occ in existing_metadata['occurrences']]
        avg_area = np.mean(existing_areas)

        area_ratio = min(new_area, avg_area) / max(new_area, avg_area)
        if area_ratio < 0.25:
            return False, f"area_mismatch(ratio={area_ratio:.2f})"

        if 'spatial_consistency' in similarity_details:
            spatial_sim = similarity_details['spatial_consistency']
            if spatial_sim < 0.3:
                return False, f"poor_spatial_match({spatial_sim:.3f})"

        if 'siglip_similarity' in similarity_details:
            siglip_sim = similarity_details['siglip_similarity']
            if siglip_sim < 0.50:
                return False, f"poor_siglip_semantic_match({siglip_sim:.3f})"

        return True, "valid"

    def _update_features_by_quality_selection(self, landmark_id, new_landmark, new_quality, existing_quality):
        """Update features based on quality selection"""
        existing_features = self.landmark_features[landmark_id]
        new_features = new_landmark['features']

        if new_quality > existing_quality:
            updated_features = {}

            for modality in ['siglip', 'dinov2']:
                if modality in new_features and new_features[modality] is not None:
                    updated_features[modality] = new_features[modality].to(self.device)
                elif modality in existing_features and existing_features[modality] is not None:
                    updated_features[modality] = existing_features[modality].to(self.device)

            self._update_feature_source_stats(landmark_id, 'new_landmark_selected', new_quality, existing_quality)

        else:
            updated_features = {}

            for modality in ['siglip', 'dinov2']:
                if modality in existing_features and existing_features[modality] is not None:
                    updated_features[modality] = existing_features[modality].to(self.device)
                elif modality in new_features and new_features[modality] is not None:
                    updated_features[modality] = new_features[modality].to(self.device)

            self._update_feature_source_stats(landmark_id, 'existing_landmark_kept', new_quality, existing_quality)

        self.landmark_features[landmark_id] = updated_features

        self.stats['feature_selection_count'] = self.stats.get('feature_selection_count', 0) + 1

    def _update_feature_source_stats(self, landmark_id, selection_type, new_quality, existing_quality):
        """Update feature selection statistics"""
        if 'feature_source_stats' not in self.stats:
            self.stats['feature_source_stats'] = {
                'new_landmark_selected': 0,
                'existing_landmark_kept': 0,
                'quality_improvements': [],
                'quality_differences': []
            }

        self.stats['feature_source_stats'][selection_type] += 1

        quality_diff = new_quality - existing_quality
        self.stats['feature_source_stats']['quality_differences'].append(quality_diff)

        if selection_type == 'new_landmark_selected':
            self.stats['feature_source_stats']['quality_improvements'].append(quality_diff)

    def _update_metadata(self, landmark_id, new_landmark, image_path, timestamp, scene_info=None):
        """Update landmark metadata"""
        metadata = self.landmark_metadata[landmark_id]

        new_occurrence = {
            'image_path': image_path,
            'timestamp': timestamp,
            'bbox': new_landmark['bbox'],
            'center': new_landmark['center'],
            'area': new_landmark['area'],
            'quality_score': new_landmark['quality_score'],
            'stability_score': new_landmark['stability_score'],
            'landmark_image_path': new_landmark.get('landmark_image_path'),
        }

        metadata['occurrences'].append(new_occurrence)
        metadata['last_seen'] = timestamp
        metadata['total_occurrences'] += 1

        all_qualities = [occ['quality_score'] for occ in metadata['occurrences']]
        all_areas = [occ['area'] for occ in metadata['occurrences']]

        metadata['average_quality'] = np.mean(all_qualities)
        metadata['average_area'] = np.mean(all_areas)

        best_occurrence = max(metadata['occurrences'], key=lambda x: x['quality_score'])
        metadata['representative_bbox'] = best_occurrence['bbox']
        metadata['representative_center'] = best_occurrence['center']
        metadata['representative_image_path'] = best_occurrence.get('landmark_image_path')

        if scene_info and scene_info.get('analysis_success', False):
            self._update_scene_observations(landmark_id, scene_info)

        if metadata['representative_image_path']:
            if landmark_id not in [lid for lid in self.landmark_metadata.keys()
                                   if self.landmark_metadata[lid].get('representative_image_path')]:
                self.stats['landmarks_with_images'] += 1

        total_images = sum(1 for metadata in self.landmark_metadata.values()
                           for occ in metadata.get('occurrences', [])
                           if occ.get('landmark_image_path'))
        self.stats['total_landmark_images'] = total_images

        self.graph.nodes[landmark_id].update(metadata)

    def _update_scene_observations(self, landmark_id: int, scene_info: dict):
        """Update landmark scene observation statistics"""
        try:
            metadata = self.landmark_metadata[landmark_id]

            if 'scene_observations' not in metadata:
                metadata['scene_observations'] = {}

            scene_observations = metadata['scene_observations']
            scene_type = scene_info.get('scene_category', 'unknown')
            confidence = scene_info.get('confidence', 0.0)

            if scene_type in scene_observations:
                scene_observations[scene_type]['count'] += 1
                scene_observations[scene_type]['total_confidence'] += confidence
            else:
                scene_observations[scene_type] = {'count': 1, 'total_confidence': confidence}

            self._update_derived_scene_info(landmark_id)

        except Exception:
            pass

    def _create_landmark(self, landmark, image_path, timestamp, scene_info=None):
        """Create new landmark"""
        landmark_id = self.global_landmark_id
        self.global_landmark_id += 1

        features = {}
        for modality, feature in landmark['features'].items():
            if feature is not None:
                features[modality] = feature.to(self.device)

        self.landmark_features[landmark_id] = features
        self.landmark_creation_info[landmark_id] = {
            'frame_index': self.current_frame_index,
            'timestamp': timestamp,
            'image_path': image_path,
            'creation_order': len(self.landmark_creation_info)
        }

        metadata = {
            'landmark_id': landmark_id,
            'first_seen': timestamp,
            'last_seen': timestamp,
            'occurrences': [{
                'image_path': image_path,
                'timestamp': timestamp,
                'bbox': landmark['bbox'],
                'center': landmark['center'],
                'area': landmark['area'],
                'quality_score': landmark['quality_score'],
                'stability_score': landmark['stability_score'],
                'landmark_image_path': landmark.get('landmark_image_path'),
            }],
            'total_occurrences': 1,
            'average_area': landmark['area'],
            'average_quality': landmark['quality_score'],
            'representative_bbox': landmark['bbox'],
            'representative_center': landmark['center'],
            'representative_image_path': landmark.get('landmark_image_path'),
        }

        metadata['scene_observations'] = {}
        metadata['scene_entropy'] = 0.0
        metadata['dominant_scene'] = 'unknown'
        metadata['is_boundary_landmark'] = False
        metadata['is_multi_scene_landmark'] = False

        if scene_info and scene_info.get('analysis_success', False):
            scene_type = scene_info.get('scene_category', 'unknown')
            confidence = scene_info.get('confidence', 0.0)

            metadata['scene_observations'][scene_type] = {
                'count': 1,
                'total_confidence': confidence
            }
            metadata['dominant_scene'] = scene_type

            metadata['scene_indoor_outdoor'] = scene_info.get('indoor_outdoor', 'unknown')
            metadata['scene_category'] = scene_type
            metadata['scene_confidence'] = confidence
        else:
            metadata['scene_indoor_outdoor'] = 'unknown'
            metadata['scene_category'] = 'unknown'
            metadata['scene_confidence'] = 0.0

        self.landmark_metadata[landmark_id] = metadata
        self.graph.add_node(landmark_id, **metadata)

        if landmark.get('landmark_image_path'):
            self.stats['landmarks_with_images'] += 1
            self.stats['total_landmark_images'] += 1

        if scene_info and scene_info.get('analysis_success', False):
            indoor_outdoor = scene_info.get('indoor_outdoor', 'unknown')
            scene_category = scene_info.get('scene_category', 'unknown')

            if indoor_outdoor == 'indoor':
                self.stats['scene_analysis_stats']['indoor_landmarks'] += 1
            elif indoor_outdoor == 'outdoor':
                self.stats['scene_analysis_stats']['outdoor_landmarks'] += 1
            else:
                self.stats['scene_analysis_stats']['unknown_scene_landmarks'] += 1

            if scene_category != 'unknown':
                self.stats['scene_analysis_stats']['scene_categories'][scene_category] += 1

        return landmark_id

    def get_landmark_temporal_info(self, landmark_id: int) -> dict:
        """Get landmark temporal information"""
        if landmark_id not in self.landmark_creation_info:
            return {'exists': False}

        creation_info = self.landmark_creation_info[landmark_id]
        current_frame = self.current_frame_index

        return {
            'exists': True,
            'creation_frame': creation_info['frame_index'],
            'creation_timestamp': creation_info['timestamp'],
            'creation_image': creation_info['image_path'],
            'frame_age': current_frame - creation_info['frame_index'],
            'creation_order': creation_info['creation_order'],
            'is_recent': (current_frame - creation_info['frame_index']) <= self.merge_temporal_window
        }

    def _update_semantic_clusters(self, landmark_ids):
        """Update semantic clusters using SigLIP features"""
        for landmark_id in landmark_ids:
            if landmark_id not in self.landmark_features:
                continue

            features = self.landmark_features[landmark_id]
            if 'siglip' not in features or features['siglip'] is None:
                continue

            siglip_features = features['siglip']

            if siglip_features.dim() > 1 and siglip_features.shape[0] > 1:
                siglip_features = siglip_features.mean(dim=0, keepdim=True)
            if siglip_features.dim() == 1:
                siglip_features = siglip_features.unsqueeze(0)

            best_cluster = None
            best_similarity = 0.0

            for cluster_id, representative_features in self.cluster_representatives.items():
                try:
                    rep_features = representative_features
                    if rep_features.dim() > 1 and rep_features.shape[0] > 1:
                        rep_features = rep_features.mean(dim=0, keepdim=True)
                    if rep_features.dim() == 1:
                        rep_features = rep_features.unsqueeze(0)

                    if hasattr(self.siglip_encoder, 'compute_enhanced_similarity'):
                        similarity = self.siglip_encoder.compute_enhanced_similarity(
                            siglip_features, rep_features
                        )
                    else:
                        similarity = torch.cosine_similarity(siglip_features, rep_features, dim=1)
                        if similarity.numel() > 1:
                            similarity = similarity.mean()
                        similarity = similarity.item()

                    if similarity > best_similarity and similarity > 0.70:
                        best_similarity = similarity
                        best_cluster = cluster_id
                except Exception:
                    continue

            if best_cluster is not None:
                self.semantic_clusters[best_cluster].append(landmark_id)
            else:
                new_cluster_id = len(self.cluster_representatives)
                self.semantic_clusters[new_cluster_id] = [landmark_id]
                self.cluster_representatives[new_cluster_id] = siglip_features

    def print_stats(self):
        """Print comprehensive statistics"""
        print(f"\nGRAPH STATISTICS")
        print("=" * 50)
        print(f"Unique landmarks: {len(self.landmark_features)}")
        print(f"Landmarks with images: {self.stats['landmarks_with_images']}")
        print(f"Total landmark images: {self.stats['total_landmark_images']}")
        print(f"Total edges: {self.graph.number_of_edges()}")
        print(f"Processed images: {self.stats['processed_images']}")
        print(f"Successful merges: {self.stats['merged_landmarks']}")
        print(f"Rejected merges: {self.stats['rejected_merges']}")

        print(f"\nTemporal Merge Control:")
        print(f"   Merge temporal window: {self.merge_temporal_window} frames")
        print(f"   Current frame index: {self.current_frame_index}")

        if self.landmark_creation_info:
            frame_ages = [self.current_frame_index - info['frame_index']
                          for info in self.landmark_creation_info.values()]
            recent_count = sum(1 for age in frame_ages if age <= self.merge_temporal_window)

            print(f"   Recent landmarks (≤{self.merge_temporal_window} frames): {recent_count}/{len(frame_ages)}")
            print(f"   Average landmark age: {np.mean(frame_ages):.1f} frames")
            print(f"   Max landmark age: {max(frame_ages) if frame_ages else 0} frames")

        total_attempts = self.stats['merged_landmarks'] + self.stats['rejected_merges']
        if total_attempts > 0:
            success_rate = self.stats['merged_landmarks'] / total_attempts
            print(f"Merge success rate: {success_rate:.1%}")

        if len(self.landmark_features) > 0:
            image_coverage = self.stats['landmarks_with_images'] / len(self.landmark_features)
            print(f"Image coverage rate: {image_coverage:.1%}")

        feature_stats = self.stats.get('feature_source_stats', {})
        if feature_stats:
            print(f"\nFeature Selection Strategy (Quality-based):")
            new_selected = feature_stats.get('new_landmark_selected', 0)
            existing_kept = feature_stats.get('existing_landmark_kept', 0)
            total_selections = new_selected + existing_kept

            if total_selections > 0:
                new_rate = new_selected / total_selections
                existing_rate = existing_kept / total_selections
                print(f"   New landmark features selected: {new_selected} ({new_rate:.1%})")
                print(f"   Existing features kept: {existing_kept} ({existing_rate:.1%})")

                quality_improvements = feature_stats.get('quality_improvements', [])
                if quality_improvements:
                    avg_improvement = np.mean(quality_improvements)
                    print(f"   Average quality improvement: +{avg_improvement:.3f}")

                quality_diffs = feature_stats.get('quality_differences', [])
                if quality_diffs:
                    avg_diff = np.mean(quality_diffs)
                    print(f"   Average quality difference: {avg_diff:+.3f}")

        print(f"\nHybrid Relationships:")
        print(f"   Co-visibility (geometric): {self.stats['covisibility_connections']}")
        print(f"   Temporal (1-5 frames): {self.stats['temporal_connections']}")
        total_connections = self.stats['covisibility_connections'] + self.stats['temporal_connections']
        if total_connections > 0:
            covis_ratio = self.stats['covisibility_connections'] / total_connections
            temporal_ratio = self.stats['temporal_connections'] / total_connections
            print(f"   Ratio - Covisibility: {covis_ratio:.1%}, Temporal: {temporal_ratio:.1%}")

        print(f"\nLanguage queries: {self.stats['language_queries']}")
        if self.stats['language_queries'] > 0:
            match_rate = self.stats['successful_language_matches'] / self.stats['language_queries']
            print(f"Query success rate: {match_rate:.1%}")

        scene_query_stats = self.stats['scene_query_stats']
        print(f"\nScene Query Statistics:")
        print(f"   Total scene queries: {scene_query_stats['total_scene_queries']}")
        print(f"   Successful scene queries: {scene_query_stats['successful_scene_queries']}")
        print(f"   Scene entry selections: {scene_query_stats['scene_entry_selections']}")
        print(f"   Boundary landmark selections: {scene_query_stats['boundary_landmark_selections']}")
        if scene_query_stats['total_scene_queries'] > 0:
            scene_success_rate = scene_query_stats['successful_scene_queries'] / scene_query_stats[
                'total_scene_queries']
            print(f"   Scene query success rate: {scene_success_rate:.1%}")

        if scene_query_stats['scene_types_queried']:
            print(f"   Top queried scene types:")
            sorted_scenes = sorted(scene_query_stats['scene_types_queried'].items(), key=lambda x: x[1], reverse=True)
            for scene_type, count in sorted_scenes[:5]:
                print(f"     {scene_type}: {count} queries")

        if scene_query_stats['scene_match_score_distribution']:
            avg_match_score = np.mean(scene_query_stats['scene_match_score_distribution'])
            print(f"   Average scene match score: {avg_match_score:.3f}")

        print(f"\nSimilarity Components:")
        components = [
            ('SigLIP', self.stats['siglip_similarity_distribution']),
            ('DINOv2', self.stats['dinov2_similarity_distribution']),
            ('Spatial', self.stats['spatial_similarity_distribution']),
            ('Scale', self.stats['scale_similarity_distribution']),
            ('Geometric Distance', self.stats['geometric_distance_distribution']),
            ('Temporal Distance', self.stats['temporal_distance_distribution']),
        ]

        for name, distribution in components:
            if distribution:
                avg_sim = np.mean(distribution)
                print(f"   {name}: {avg_sim:.3f} (n={len(distribution)})")

        scene_stats = self.stats['scene_analysis_stats']
        total_landmarks = len(self.landmark_features)
        if total_landmarks > 0:
            print(f"\nScene Analysis Statistics:")
            print(
                f"   Indoor landmarks: {scene_stats['indoor_landmarks']} ({scene_stats['indoor_landmarks'] / total_landmarks:.1%})")
            print(
                f"   Outdoor landmarks: {scene_stats['outdoor_landmarks']} ({scene_stats['outdoor_landmarks'] / total_landmarks:.1%})")
            print(
                f"   Unknown scene landmarks: {scene_stats['unknown_scene_landmarks']} ({scene_stats['unknown_scene_landmarks'] / total_landmarks:.1%})")
            print(
                f"   Boundary landmarks: {scene_stats['boundary_landmarks']} ({scene_stats['boundary_landmarks'] / total_landmarks:.1%})")
            print(
                f"   Multi-scene landmarks: {scene_stats['multi_scene_landmarks']} ({scene_stats['multi_scene_landmarks'] / total_landmarks:.1%})")

            if scene_stats['scene_categories']:
                print(f"   Top scene categories:")
                sorted_categories = sorted(scene_stats['scene_categories'].items(), key=lambda x: x[1], reverse=True)
                for category, count in sorted_categories[:5]:
                    print(f"     {category}: {count} landmarks")

        landmark_metrics = []
        for lid, meta in self.landmark_metadata.items():
            dominant_scene = meta.get('dominant_scene', 'unknown')
            is_boundary = meta.get('is_boundary_landmark', False)
            scene_entropy = meta.get('scene_entropy', 0.0)

            landmark_metrics.append({
                'id': lid,
                'occurrences': meta['total_occurrences'],
                'quality': meta['average_quality'],
                'has_image': meta.get('representative_image_path') is not None,
                'dominant_scene': dominant_scene,
                'is_boundary': is_boundary,
                'scene_entropy': scene_entropy,
            })

        landmark_metrics.sort(key=lambda x: x['occurrences'], reverse=True)

        print(f"\nTop landmarks:")
        for i, lm in enumerate(landmark_metrics[:3]):
            image_status = "[IMG]" if lm['has_image'] else "[NO-IMG]"
            boundary_status = "[BOUNDARY]" if lm['is_boundary'] else ""
            scene_info = f"{lm['dominant_scene']} (entropy:{lm['scene_entropy']:.2f})"
            print(
                f"   {i + 1}. L{lm['id']}: {lm['occurrences']} occ, quality {lm['quality']:.3f} {image_status}{boundary_status} [{scene_info}]")

    def save_graph(self, save_path: str):
        """Save graph data"""
        try:
            serializable_features = {}
            for landmark_id, features in self.landmark_features.items():
                serializable_features[str(landmark_id)] = {}
                for modality, feature_tensor in features.items():
                    if feature_tensor is not None:
                        serializable_features[str(landmark_id)][modality] = feature_tensor.cpu().tolist()

            serializable_metadata = {}
            for landmark_id, metadata in self.landmark_metadata.items():
                serializable_metadata[str(landmark_id)] = {}
                for key, value in metadata.items():
                    if key != 'occurrences':
                        serializable_metadata[str(landmark_id)][key] = value
                    else:
                        occurrences = []
                        for occ in value:
                            occ_copy = {k: v for k, v in occ.items()
                                        if isinstance(v, (int, float, str, list))}
                            occurrences.append(occ_copy)
                        serializable_metadata[str(landmark_id)]['occurrences'] = occurrences

            serializable_stats = dict(self.stats)
            if 'merge_rejection_reasons' in serializable_stats:
                serializable_stats['merge_rejection_reasons'] = dict(serializable_stats['merge_rejection_reasons'])
            if 'scene_analysis_stats' in serializable_stats:
                serializable_stats['scene_analysis_stats'] = dict(serializable_stats['scene_analysis_stats'])
                if 'scene_categories' in serializable_stats['scene_analysis_stats']:
                    serializable_stats['scene_analysis_stats']['scene_categories'] = dict(
                        serializable_stats['scene_analysis_stats']['scene_categories']
                    )
            if 'scene_query_stats' in serializable_stats:
                serializable_stats['scene_query_stats'] = dict(serializable_stats['scene_query_stats'])
                if 'scene_types_queried' in serializable_stats['scene_query_stats']:
                    serializable_stats['scene_query_stats']['scene_types_queried'] = dict(
                        serializable_stats['scene_query_stats']['scene_types_queried']
                    )

            save_data = {
                'graph': nx.node_link_data(self.graph),
                'landmark_features': serializable_features,
                'landmark_metadata': serializable_metadata,
                'image_landmarks': self.image_landmarks,
                'semantic_clusters': {str(k): v for k, v in self.semantic_clusters.items()},
                'learned_thresholds': self.learned_thresholds,
                'stats': serializable_stats,
                'timestamp': datetime.now().isoformat(),
                'feature_alignment': 'siglip_aligned',
                'image_storage_enabled': True,
                'hybrid_relationships_enabled': True,
                'covisibility_method': 'geometric_distance_based',
                'temporal_method': 'frame_difference_based',
                'temporal_window_frames': self.temporal_window_frames,
                'scene_aware_landmarks': True,
                'mixed_target_parsing_enabled': True,
                'optimized_scene_info_enabled': True,
                'scene_probability_fusion': True,
            }

            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Save failed: {e}")

    def load_graph(self, load_path: str):
        """Load graph data"""
        try:
            load_path = Path(load_path)
            if not load_path.exists():
                raise FileNotFoundError(f"Graph file not found: {load_path}")

            with open(load_path, 'r') as f:
                load_data = json.load(f)

            print(f"Loading graph from: {load_path}")

            if 'graph' in load_data:
                self.graph = nx.node_link_graph(load_data['graph'])
                print(f"   Graph structure loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

            if 'landmark_features' in load_data:
                self.landmark_features = {}
                for landmark_id_str, features_dict in load_data['landmark_features'].items():
                    landmark_id = int(landmark_id_str)
                    self.landmark_features[landmark_id] = {}

                    for modality, feature_list in features_dict.items():
                        if feature_list is not None:
                            feature_tensor = torch.tensor(feature_list, dtype=torch.float32)
                            self.landmark_features[landmark_id][modality] = feature_tensor.to(self.device)

                print(f"   Landmark features loaded: {len(self.landmark_features)} landmarks")

            if 'landmark_metadata' in load_data:
                self.landmark_metadata = {}
                for landmark_id_str, metadata in load_data['landmark_metadata'].items():
                    landmark_id = int(landmark_id_str)
                    self.landmark_metadata[landmark_id] = metadata

                print(f"   Landmark metadata loaded: {len(self.landmark_metadata)} landmarks")

            if 'image_landmarks' in load_data:
                self.image_landmarks = load_data['image_landmarks']
                print(f"   Image landmarks mapping loaded: {len(self.image_landmarks)} images")

            if 'semantic_clusters' in load_data:
                self.semantic_clusters = {}
                for cluster_id_str, landmark_ids in load_data['semantic_clusters'].items():
                    if cluster_id_str.isdigit():
                        cluster_id = int(cluster_id_str)
                        self.semantic_clusters[cluster_id] = landmark_ids

            if 'learned_thresholds' in load_data:
                self.learned_thresholds.update(load_data['learned_thresholds'])

            if 'stats' in load_data:
                self.stats.update(load_data['stats'])
                if 'merge_rejection_reasons' in self.stats:
                    self.stats['merge_rejection_reasons'] = defaultdict(int, self.stats['merge_rejection_reasons'])

                if 'scene_analysis_stats' in self.stats:
                    if 'scene_categories' in self.stats['scene_analysis_stats']:
                        self.stats['scene_analysis_stats']['scene_categories'] = defaultdict(
                            int, self.stats['scene_analysis_stats']['scene_categories']
                        )

                if 'scene_query_stats' in self.stats:
                    if 'scene_types_queried' in self.stats['scene_query_stats']:
                        self.stats['scene_query_stats']['scene_types_queried'] = defaultdict(
                            int, self.stats['scene_query_stats']['scene_types_queried']
                        )

            if 'temporal_window_frames' in load_data:
                self.temporal_window_frames = load_data['temporal_window_frames']

            if self.landmark_metadata:
                self.global_landmark_id = max(self.landmark_metadata.keys()) + 1

            self._validate_loaded_data()

            print(f"   Graph loaded successfully!")
            print(f"   Features: {len(self.landmark_features)} landmarks")
            print(f"   Metadata: {len(self.landmark_metadata)} landmarks")
            print(f"   Images: {len(self.image_landmarks)} mapped")

            landmarks_with_images = self.stats.get('landmarks_with_images', 0)
            scene_analysis_enabled = load_data.get('scene_aware_landmarks', False)
            print(f"   Landmarks w/ images: {landmarks_with_images}")
            print(f"   Scene analysis: {'Enabled' if scene_analysis_enabled else 'Disabled'}")

        except Exception as e:
            print(f"Failed to load graph: {e}")
            raise e

    def _validate_loaded_data(self):
        """Validate loaded data integrity"""
        try:
            feature_ids = set(self.landmark_features.keys())
            metadata_ids = set(self.landmark_metadata.keys())
            graph_node_ids = set(self.graph.nodes())

            if feature_ids != metadata_ids:
                print(f"Warning: Feature IDs ({len(feature_ids)}) != Metadata IDs ({len(metadata_ids)})")

            if feature_ids != graph_node_ids:
                print(f"Warning: Feature IDs ({len(feature_ids)}) != Graph node IDs ({len(graph_node_ids)})")

            all_image_landmark_ids = set()
            for image_path, landmark_ids in self.image_landmarks.items():
                all_image_landmark_ids.update(landmark_ids)

            missing_in_features = all_image_landmark_ids - feature_ids
            if missing_in_features:
                print(f"Warning: {len(missing_in_features)} landmark IDs in image mapping not found in features")

            print("Data validation completed")

        except Exception:
            pass

    def visualize_graph(self, save_path: str = None, figsize=(15, 10)):
        """Visualize graph with comprehensive statistics"""
        if len(self.graph.nodes()) == 0:
            return

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

            node_sizes = [self.landmark_metadata[node]['total_occurrences'] * 50
                          for node in self.graph.nodes()]

            node_colors = []
            for node in self.graph.nodes():
                metadata = self.landmark_metadata[node]
                entropy = metadata.get('scene_entropy', 0.0)
                if entropy > 0:
                    node_colors.append(entropy)
                else:
                    node_colors.append(metadata['average_quality'])

            nx.draw_networkx_nodes(self.graph, pos, ax=ax1,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap='viridis', alpha=0.8)

            covisibility_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('type') == 'covisibility']
            temporal_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('type') == 'temporal']

            if covisibility_edges:
                nx.draw_networkx_edges(self.graph, pos, edgelist=covisibility_edges,
                                       edge_color='blue', alpha=0.8, ax=ax1, width=2, label='Covisibility')

            if temporal_edges:
                nx.draw_networkx_edges(self.graph, pos, edgelist=temporal_edges,
                                       edge_color='orange', alpha=0.5, ax=ax1, width=1, style='dashed',
                                       label='Temporal')

            ax1.set_title('Hybrid Landmark Graph (SigLIP + Scene Probability)\nSize: occurrences, Color: scene entropy/quality')
            ax1.legend()

            qualities = [meta['average_quality'] for meta in self.landmark_metadata.values()]
            if qualities:
                ax2.hist(qualities, bins=20, alpha=0.7, color='green', label='Quality')

            entropies = [meta.get('scene_entropy', 0.0) for meta in self.landmark_metadata.values() if
                         meta.get('scene_entropy', 0.0) > 0]
            if entropies:
                ax2.hist(entropies, bins=20, alpha=0.5, color='red', label='Scene Entropy')

            ax2.set_title('Quality & Scene Entropy Distribution')
            ax2.set_xlabel('Score')
            ax2.legend()

            bins = np.linspace(0, 1, 20)
            has_data = False

            if self.stats.get('siglip_similarity_distribution'):
                ax3.hist(self.stats['siglip_similarity_distribution'],
                         bins=bins, alpha=0.6, color='red', label='SigLIP')
                has_data = True
            if self.stats.get('dinov2_similarity_distribution'):
                ax3.hist(self.stats['dinov2_similarity_distribution'],
                         bins=bins, alpha=0.6, color='orange', label='DINOv2')
                has_data = True
            if self.stats.get('geometric_distance_distribution'):
                ax3.hist(self.stats['geometric_distance_distribution'],
                         bins=bins, alpha=0.6, color='blue', label='Geo Distance')
                has_data = True
            if self.stats.get('temporal_distance_distribution'):
                ax3.hist(self.stats['temporal_distance_distribution'],
                         bins=bins, alpha=0.6, color='purple', label='Temporal Distance')
                has_data = True
            if self.stats['scene_query_stats'].get('scene_match_score_distribution'):
                ax3.hist(self.stats['scene_query_stats']['scene_match_score_distribution'],
                         bins=bins, alpha=0.4, color='green', label='Scene Match')
                has_data = True

            if has_data:
                ax3.set_title('Feature Similarity & Scene Matching Distribution')
                ax3.set_xlabel('Score/Distance')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No similarity data', transform=ax3.transAxes, ha='center')

            ax4.axis('off')
            total_merges = self.stats['merged_landmarks'] + self.stats['rejected_merges']
            success_rate = self.stats['merged_landmarks'] / max(total_merges, 1)
            image_coverage = self.stats['landmarks_with_images'] / max(len(self.landmark_features), 1)

            geo_dist_stats = ""
            if self.stats['geometric_distance_distribution']:
                avg_geo_dist = np.mean(self.stats['geometric_distance_distribution'])
                geo_dist_stats = f"• Avg geometric distance: {avg_geo_dist:.3f}\n"

            temporal_dist_stats = ""
            if self.stats['temporal_distance_distribution']:
                avg_temporal_dist = np.mean(self.stats['temporal_distance_distribution'])
                temporal_dist_stats = f"• Avg temporal distance: {avg_temporal_dist:.3f}\n"

            scene_stats = self.stats['scene_analysis_stats']
            total_landmarks = len(self.landmark_features)
            scene_stats_text = ""
            if total_landmarks > 0:
                indoor_pct = scene_stats['indoor_landmarks'] / total_landmarks * 100
                outdoor_pct = scene_stats['outdoor_landmarks'] / total_landmarks * 100
                unknown_pct = scene_stats['unknown_scene_landmarks'] / total_landmarks * 100
                boundary_pct = scene_stats['boundary_landmarks'] / total_landmarks * 100
                multi_scene_pct = scene_stats['multi_scene_landmarks'] / total_landmarks * 100

                scene_stats_text = f"""
Scene Distribution:
• Indoor: {scene_stats['indoor_landmarks']} ({indoor_pct:.1f}%)
• Outdoor: {scene_stats['outdoor_landmarks']} ({outdoor_pct:.1f}%)
• Unknown: {scene_stats['unknown_scene_landmarks']} ({unknown_pct:.1f}%)
• Boundary: {scene_stats['boundary_landmarks']} ({boundary_pct:.1f}%)
• Multi-scene: {scene_stats['multi_scene_landmarks']} ({multi_scene_pct:.1f}%)
"""

            scene_query_stats = self.stats['scene_query_stats']
            scene_query_success_rate = 0
            if scene_query_stats['total_scene_queries'] > 0:
                scene_query_success_rate = scene_query_stats['successful_scene_queries'] / scene_query_stats[
                    'total_scene_queries'] * 100

            scene_query_text = f"""
Mixed Target Parsing:
• Scene queries: {scene_query_stats['total_scene_queries']}
• Success rate: {scene_query_success_rate:.1f}%
• Entry selections: {scene_query_stats['scene_entry_selections']}
• Boundary selections: {scene_query_stats['boundary_landmark_selections']}
"""

            total_connections = self.stats['covisibility_connections'] + self.stats['temporal_connections']
            covis_ratio = self.stats['covisibility_connections'] / max(total_connections, 1) * 100
            temporal_ratio = self.stats['temporal_connections'] / max(total_connections, 1) * 100

            stats_text = f"""
SigLIP Navigation System

Feature Alignment: SigLIP ALIGNED
Relationship Model: HYBRID (Covis + Temporal)
Scene Info Model: PROBABILITY FUSION
Temporal Window: {self.temporal_window_frames} frames
Image Storage: ENABLED
Scene Analysis: OPTIMIZED

Landmarks: {len(self.landmark_features)}
Landmarks w/ Images: {self.stats['landmarks_with_images']}
Image Coverage: {image_coverage:.1%}
Total Images: {self.stats['total_landmark_images']}
Processed Views: {self.stats['processed_images']}

Hybrid Relationships:
• Covisibility: {self.stats['covisibility_connections']} ({covis_ratio:.1f}%)
• Temporal: {self.stats['temporal_connections']} ({temporal_ratio:.1f}%)
• Total Edges: {self.graph.number_of_edges()}

Merge Performance:
• Success rate: {success_rate:.1%}
• Successful: {self.stats['merged_landmarks']}
• Rejected: {self.stats['rejected_merges']}

Language Queries: {self.stats['language_queries']}
Query Success: {self.stats['successful_language_matches']}

{geo_dist_stats}{temporal_dist_stats}
{scene_stats_text}
{scene_query_text}
"""

            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')

            plt.close()

        except Exception as e:
            print(f"Visualization failed: {e}")

    def get_landmark_scene_probability_distribution(self, landmark_id: int) -> Dict[str, float]:
        """Get scene probability distribution for landmark"""
        try:
            metadata = self.landmark_metadata.get(landmark_id)
            if not metadata:
                return {}

            scene_observations = metadata.get('scene_observations', {})
            if not scene_observations:
                dominant_scene = metadata.get('scene_category', 'unknown')
                confidence = metadata.get('scene_confidence', 0.0)
                if dominant_scene != 'unknown':
                    return {dominant_scene: confidence}
                return {}

            total_count = sum(obs['count'] for obs in scene_observations.values())
            if total_count == 0:
                return {}

            probability_distribution = {}
            for scene, obs in scene_observations.items():
                probability = obs['count'] / total_count
                avg_confidence = obs['total_confidence'] / obs['count']
                final_probability = probability * avg_confidence
                probability_distribution[scene] = final_probability

            return probability_distribution

        except Exception:
            return {}

    def get_boundary_landmarks(self) -> List[int]:
        """Get all boundary landmarks"""
        try:
            boundary_landmarks = []
            for landmark_id, metadata in self.landmark_metadata.items():
                if metadata.get('is_boundary_landmark', False):
                    boundary_landmarks.append(landmark_id)
            return boundary_landmarks
        except Exception:
            return []

    def get_multi_scene_landmarks(self) -> List[int]:
        """Get all multi-scene landmarks"""
        try:
            multi_scene_landmarks = []
            for landmark_id, metadata in self.landmark_metadata.items():
                if metadata.get('is_multi_scene_landmark', False):
                    multi_scene_landmarks.append(landmark_id)
            return multi_scene_landmarks
        except Exception:
            return []
