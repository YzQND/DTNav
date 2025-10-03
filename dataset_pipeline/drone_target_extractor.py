#!/usr/bin/env python3
"""
3D Drone Dataset Target Extractor - Optimized Version.
Based on efficient processing strategies: model sharing, batch preprocessing, caching mechanism.
"""

import json
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import sys
from tqdm import tqdm
import torch
import shutil
from collections import defaultdict
import cv2
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import random


def sanitize_trajectory_id(trajectory_id: str) -> str:
    """Sanitize trajectory ID by replacing unsafe path characters"""
    safe_id = trajectory_id.replace('/', '_').replace('\\', '_')
    safe_id = re.sub(r'[<>:"|?*]', '_', safe_id)
    return safe_id


from navigation_system import LandmarkNavigationSystem
from landmark_extractor import LandmarkExtractor
from utils import setup_environment, get_best_gpu, get_safe_output_dir


@dataclass
class Drone3DTargetPair:
    """3D drone target pair data structure"""
    trajectory_id: str
    scene_id: str
    current_frame_id: str
    navigation_frame_id: str
    current_action: str
    visible_target_id: Optional[int]
    visible_target_path: Optional[str]
    visible_target_confidence: float
    navigation_target_id: Optional[int]
    navigation_target_path: Optional[str]
    navigation_target_confidence: float
    current_landmarks_count: int
    navigation_landmarks_count: int
    frame_distance: int
    pair_type: str


@dataclass
class Drone3DLandmarkInfo:
    """3D drone landmark information"""
    landmark_id: int
    trajectory_id: str
    image_path: str
    quality_score: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: float


class Drone3DLandmarkLibrary:
    """Trajectory-specific landmark library manager for 3D drone"""

    def __init__(self, base_output_dir: Path, trajectory_id: str, verbose: bool = False):
        self.trajectory_id = trajectory_id
        safe_trajectory_id = sanitize_trajectory_id(trajectory_id)
        self.base_dir = base_output_dir / "drone_3d_landmark_library" / f"trajectory_{safe_trajectory_id}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.landmark_registry = {}
        self.stats = {
            'trajectory_id': trajectory_id,
            'safe_trajectory_id': safe_trajectory_id,
            'total_landmarks': 0,
            'successful_saves': 0,
            'failed_saves': 0,
        }

    def register_landmark(self, landmark_id: int, landmark_info: Drone3DLandmarkInfo) -> bool:
        """Register landmark to trajectory-specific library"""
        if landmark_id in self.landmark_registry:
            return True

        try:
            landmark_image_path = self._save_landmark_image(landmark_id, landmark_info)

            if landmark_image_path:
                landmark_info.image_path = landmark_image_path
                self.landmark_registry[landmark_id] = landmark_info
                self.stats['successful_saves'] += 1
                self.stats['total_landmarks'] += 1
                return True
            else:
                self.stats['failed_saves'] += 1
                return False

        except Exception as e:
            if self.verbose:
                print(f"Failed to register landmark {landmark_id} for trajectory {self.trajectory_id}: {e}")
            self.stats['failed_saves'] += 1
            return False

    def _save_landmark_image(self, landmark_id: int, landmark_info: Drone3DLandmarkInfo) -> Optional[str]:
        """Save landmark representative image"""
        try:
            safe_trajectory_id = sanitize_trajectory_id(self.trajectory_id)
            target_filename = f"traj_{safe_trajectory_id}_landmark_{landmark_id:04d}.png"
            target_path = self.base_dir / target_filename

            image_path = landmark_info.image_path
            bbox = landmark_info.bbox

            image = cv2.imread(image_path)
            if image is None:
                return None

            x, y, w, h = bbox
            x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)

            h_img, w_img = image.shape[:2]
            x = min(x, w_img - 1)
            y = min(y, h_img - 1)
            w = min(w, w_img - x)
            h = min(h, h_img - y)

            cropped = image[y:y + h, x:x + w]

            if cropped.size > 0:
                cv2.imwrite(str(target_path), cropped)
                return str(target_path)

        except Exception as e:
            if self.verbose:
                print(f"Failed to save landmark {landmark_id} image for trajectory {self.trajectory_id}: {e}")

        return None

    def get_landmark_info(self, landmark_id: int) -> Optional[Drone3DLandmarkInfo]:
        """Get landmark information"""
        return self.landmark_registry.get(landmark_id)

    def get_landmark_image_path(self, landmark_id: int) -> Optional[str]:
        """Get landmark image path"""
        landmark_info = self.landmark_registry.get(landmark_id)
        if landmark_info and Path(landmark_info.image_path).exists():
            return landmark_info.image_path
        return None


class Drone3DTargetExtractor:
    """3D drone target extractor - optimized version"""

    def __init__(self,
                 drone_base_path: str,
                 frame_pairs_file: str,
                 fastsam_model_path: str = "FastSAM-x.pt",
                 siglip_model_path: str = "google/siglip-so400m-patch14-384",
                 dinov2_model_path: str = None,
                 llava_model_path: str = None,
                 preferred_gpu_id: int = None,
                 verbose: bool = False,
                 enable_visualization: bool = False):
        """Initialize optimized 3D drone target extractor"""
        self.drone_base_path = Path(drone_base_path)
        self.frame_pairs_file = frame_pairs_file
        self.verbose = verbose
        self.enable_visualization = enable_visualization

        if preferred_gpu_id is None:
            gpu_id, gpu_name, gpu_memory = get_best_gpu()
        else:
            gpu_id = preferred_gpu_id
        self.gpu_id = gpu_id

        self.action_names = {
            None: 'start',
            'move_forward': 'move_forward',
            'turn_left': 'turn_left',
            'turn_right': 'turn_right',
            'move_up': 'move_up',
            'move_down': 'move_down',
            'stop': 'stop'
        }

        self.turn_actions = {'turn_left', 'turn_right', 'move_up', 'move_down'}

        if self.verbose:
            print("Creating shared landmark extractor instance (load models once)...")
        self.shared_landmark_extractor = LandmarkExtractor(
            fastsam_model_path=fastsam_model_path,
            siglip_model_path=siglip_model_path,
            dinov2_model=dinov2_model_path,
            preferred_gpu_id=gpu_id
        )
        if self.verbose:
            print("Shared landmark extractor created successfully")

        self.image_landmarks_cache = {}
        self.processed_images = set()

        self.trajectory_nav_systems = {}
        self.trajectory_landmark_libraries = {}

        self.stats = {
            'total_trajectories': 0,
            'total_frame_pairs': 0,
            'processed_pairs': 0,
            'valid_target_pairs': 0,
            'no_landmarks_pairs': 0,
            'failed_pairs': 0,
            'discarded_navigation_missing': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'unique_images_processed': 0,
            'model_loading_optimized': True,
            'shared_model_instances': 1,
            'action_distribution': defaultdict(int)
        }

        self.setup_logging()

        if self.verbose:
            print("Optimized 3D drone target extractor initialized (model sharing + caching)")

    def setup_logging(self):
        """Setup logging configuration"""
        handlers = [logging.FileHandler('drone_3d_target_extraction.log', encoding='utf-8')]

        if self.verbose:
            handlers.append(logging.StreamHandler(sys.stdout))

        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def load_frame_pairs(self) -> Dict:
        """Load frame pairs data"""
        try:
            with open(self.frame_pairs_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if self.verbose:
                print(f"Loaded frame pairs: {self.frame_pairs_file}")
                print(f"   Total trajectories: {data.get('total_trajectories', 0)}")
                print(f"   Total pairs: {data.get('total_pairs', 0)}")

            return data
        except Exception as e:
            print(f"Failed to load frame pairs: {e}")
            return None

    def get_image_path(self, path_id: str, frame_id: str) -> Path:
        """Get complete image path"""
        if '_' in frame_id:
            frame_idx = frame_id.split('_')[-1]
        else:
            frame_idx = frame_id.replace('.jpg', '').replace('.png', '')
        
        try:
            frame_num = int(frame_idx)
            image_filename = f"{frame_num:04d}_rgb.png"
        except ValueError:
            image_filename = frame_id if frame_id.endswith(('.jpg', '.png')) else f"{frame_id}.png"
        
        return self.drone_base_path / path_id / image_filename

    def get_landmarks_cached(self, image_path: str) -> List[Dict]:
        """Cached landmark extraction"""
        if image_path in self.image_landmarks_cache:
            self.stats['cache_hits'] += 1
            return self.image_landmarks_cache[image_path]

        self.stats['cache_misses'] += 1
        try:
            landmarks, _ = self.shared_landmark_extractor.extract_landmarks(image_path)
            self.image_landmarks_cache[image_path] = landmarks if landmarks else []
            self.processed_images.add(image_path)
            self.stats['unique_images_processed'] += 1
            return self.image_landmarks_cache[image_path]
        except Exception as e:
            self.logger.error(f"Landmark extraction failed: {image_path} - {e}")
            self.image_landmarks_cache[image_path] = []
            return []

    def create_trajectory_navigation_system(self, trajectory_id: str) -> LandmarkNavigationSystem:
        """Create independent navigation system for trajectory (using shared model)"""
        if trajectory_id in self.trajectory_nav_systems:
            return self.trajectory_nav_systems[trajectory_id]

        if self.verbose:
            print(f"Creating independent navigation system for trajectory {trajectory_id} (using shared model)...")

        nav_system = LandmarkNavigationSystem(
            fastsam_model_path="",
            siglip_model_path="",
            dinov2_model="",
            use_fastsam=True,
            base_similarity_threshold=0.75,
            preferred_gpu_id=self.gpu_id,
            shared_landmark_extractor=self.shared_landmark_extractor
        )

        self.trajectory_nav_systems[trajectory_id] = nav_system
        if self.verbose:
            print(f"Navigation system for trajectory {trajectory_id} created (using shared model)")

        return nav_system

    def create_trajectory_landmark_library(self, trajectory_id: str, output_dir: Path) -> Drone3DLandmarkLibrary:
        """Create independent landmark library for trajectory"""
        if trajectory_id in self.trajectory_landmark_libraries:
            return self.trajectory_landmark_libraries[trajectory_id]

        landmark_library = Drone3DLandmarkLibrary(output_dir, trajectory_id, self.verbose)
        self.trajectory_landmark_libraries[trajectory_id] = landmark_library
        if self.verbose:
            print(f"Landmark library for trajectory {trajectory_id} created: {landmark_library.base_dir}")

        return landmark_library

    def preprocess_trajectory_images(self, trajectory: Dict) -> Dict[str, List[Dict]]:
        """Trajectory-level batch preprocessing"""
        if self.verbose:
            print(f"Batch preprocessing trajectory images: {trajectory['path_id']}")

        unique_images = set()
        for pair in trajectory['frame_pairs']:
            current_image_path = str(self.get_image_path(trajectory['path_id'], pair['current_frame_id']))
            navigation_image_path = str(self.get_image_path(trajectory['path_id'], pair['navigation_frame_id']))
            unique_images.add(current_image_path)
            unique_images.add(navigation_image_path)

        trajectory_landmarks = {}
        for image_path in unique_images:
            if Path(image_path).exists():
                landmarks = self.get_landmarks_cached(image_path)
                trajectory_landmarks[image_path] = landmarks
            else:
                trajectory_landmarks[image_path] = []

        if self.verbose:
            total_landmarks = sum(len(landmarks) for landmarks in trajectory_landmarks.values())
            print(f"Preprocessing complete: {len(unique_images)} images, {total_landmarks} landmarks")

        return trajectory_landmarks

    def select_salient_landmark(self, landmarks: List[Dict], image_shape: Tuple[int, int] = (1920, 1080)) -> Tuple[Optional[int], float]:
        """Select salient landmark (optimized version)"""
        if not landmarks:
            return None, 0.0

        best_landmark_idx = None
        best_score = -1.0

        for idx, landmark in enumerate(landmarks):
            quality_score = landmark.get('quality_score', 0)

            area = landmark.get('area', 0)
            if area < 5000:
                size_score = area / 5000
            elif area < 50000:
                size_score = 1.0
            else:
                size_score = max(0.3, 1.0 - (area - 50000) / 100000)

            center = landmark.get('center', [image_shape[0] // 2, image_shape[1] // 2])
            image_center = (image_shape[0] // 2, image_shape[1] // 2)
            dx = abs(center[0] - image_center[0]) / (image_shape[0] / 2)
            dy = abs(center[1] - image_center[1]) / (image_shape[1] / 2)
            distance_from_center = np.sqrt(dx ** 2 + dy ** 2)
            center_score = max(0.0, 1.0 - distance_from_center)

            combined_score = 0.5 * quality_score + 0.3 * size_score + 0.2 * center_score

            if combined_score > best_score:
                best_score = combined_score
                best_landmark_idx = idx

        return best_landmark_idx, best_score

    def _find_closest_visible_target(self, current_landmarks: List[Dict], current_image_path: str,
                                     landmark_id_map: Dict[str, int], navigation_target_id: int,
                                     nav_system) -> Tuple[Optional[int], float]:
        """Find closest visible target based on navigation target and topological graph"""
        try:
            current_landmark_ids = []
            for i, landmark in enumerate(current_landmarks):
                key = f"{current_image_path}_{i}"
                landmark_id = landmark_id_map.get(key)
                if landmark_id is not None:
                    quality_score = landmark.get('quality_score', 0.0)
                    current_landmark_ids.append((landmark_id, quality_score))

            if not current_landmark_ids:
                return None, 0.0

            matched_landmarks = current_landmark_ids

            best_start_landmark, shortest_path = nav_system.topological_graph.find_shortest_path_from_current_to_target(
                matched_landmarks, navigation_target_id
            )

            if best_start_landmark is not None:
                try:
                    import networkx as nx
                    geometric_distance = nx.shortest_path_length(
                        nav_system.topological_graph.graph,
                        best_start_landmark,
                        navigation_target_id,
                        weight='geometric_distance'
                    )

                    quality_score = 0.0
                    for landmark_id, quality in current_landmark_ids:
                        if landmark_id == best_start_landmark:
                            quality_score = quality
                            break

                    distance_factor = max(0.1, 1.0 - min(1.0, geometric_distance / 2.0))
                    confidence = distance_factor * 0.7 + quality_score * 0.3

                    return best_start_landmark, confidence

                except Exception:
                    for landmark_id, quality in current_landmark_ids:
                        if landmark_id == best_start_landmark:
                            return best_start_landmark, quality
                    return best_start_landmark, 0.5

            return None, 0.0

        except Exception as e:
            if self.verbose:
                print(f"Failed to find closest visible target: {e}")
            return None, 0.0

    def visualize_trajectory_targets(self, trajectory_pairs: List[Drone3DTargetPair], trajectory_id: str, output_dir: Path):
        """Visualize trajectory target pairs (select 6 valid frame pairs for display)"""
        if not self.enable_visualization or not trajectory_pairs:
            return

        try:
            valid_pairs = [pair for pair in trajectory_pairs
                           if pair.visible_target_id is not None and pair.navigation_target_id is not None]

            if not valid_pairs:
                valid_pairs = [pair for pair in trajectory_pairs
                               if pair.visible_target_id is not None or pair.navigation_target_id is not None]

            if not valid_pairs:
                valid_pairs = trajectory_pairs

            sample_size = min(6, len(valid_pairs))
            selected_pairs = random.sample(valid_pairs, sample_size)

            fig = plt.figure(figsize=(20, 4 * sample_size))
            gs = GridSpec(sample_size, 4, figure=fig, hspace=0.3, wspace=0.2)

            for idx, pair in enumerate(selected_pairs):
                current_image_path = str(self.get_image_path(pair.trajectory_id, pair.current_frame_id))
                navigation_image_path = str(self.get_image_path(pair.trajectory_id, pair.navigation_frame_id))

                ax1 = fig.add_subplot(gs[idx, 0])
                if Path(current_image_path).exists():
                    current_img = cv2.imread(current_image_path)
                    if current_img is not None:
                        current_img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                        ax1.imshow(current_img_rgb)
                    ax1.set_title(
                        f'Current Frame\n{pair.current_frame_id}\nAction: {self.action_names.get(pair.current_action, "unknown")}',
                        fontsize=10)
                else:
                    ax1.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title(f'Current Frame (Missing)\n{pair.current_frame_id}', fontsize=10)
                ax1.axis('off')

                ax2 = fig.add_subplot(gs[idx, 1])
                if pair.visible_target_path and Path(pair.visible_target_path).exists():
                    target_img = cv2.imread(pair.visible_target_path)
                    if target_img is not None:
                        target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                        ax2.imshow(target_img_rgb)
                    ax2.set_title(
                        f'Visible Target\nID: {pair.visible_target_id}\nConf: {pair.visible_target_confidence:.3f}',
                        fontsize=10)
                else:
                    ax2.text(0.5, 0.5, 'No Visible\nTarget', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title(f'Visible Target\n(None)', fontsize=10)
                ax2.axis('off')

                ax3 = fig.add_subplot(gs[idx, 2])
                if Path(navigation_image_path).exists():
                    nav_img = cv2.imread(navigation_image_path)
                    if nav_img is not None:
                        nav_img_rgb = cv2.cvtColor(nav_img, cv2.COLOR_BGR2RGB)
                        ax3.imshow(nav_img_rgb)
                    ax3.set_title(f'Navigation Frame\n{pair.navigation_frame_id}\nDistance: {pair.frame_distance}',
                                  fontsize=10)
                else:
                    ax3.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title(f'Navigation Frame (Missing)\n{pair.navigation_frame_id}', fontsize=10)
                ax3.axis('off')

                ax4 = fig.add_subplot(gs[idx, 3])
                if pair.navigation_target_path and Path(pair.navigation_target_path).exists():
                    nav_target_img = cv2.imread(pair.navigation_target_path)
                    if nav_target_img is not None:
                        nav_target_img_rgb = cv2.cvtColor(nav_target_img, cv2.COLOR_BGR2RGB)
                        ax4.imshow(nav_target_img_rgb)
                    ax4.set_title(
                        f'Navigation Target\nID: {pair.navigation_target_id}\nConf: {pair.navigation_target_confidence:.3f}',
                        fontsize=10)
                else:
                    ax4.text(0.5, 0.5, 'No Navigation\nTarget', ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title(f'Navigation Target\n(None)', fontsize=10)
                ax4.axis('off')

            safe_trajectory_id = sanitize_trajectory_id(trajectory_id)
            fig.suptitle(f'3D Drone Trajectory: {safe_trajectory_id}\nScene: {selected_pairs[0].scene_id}',
                         fontsize=14, y=0.98)

            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            viz_filename = f"drone_trajectory_{safe_trajectory_id}_targets_visualization.png"
            viz_path = viz_dir / viz_filename

            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            if self.verbose:
                print(f"Trajectory {trajectory_id} visualization saved to: {viz_path}")

        except Exception as e:
            if self.verbose:
                print(f"Trajectory {trajectory_id} visualization failed: {e}")

    def process_trajectory_batch(self, trajectory: Dict, output_dir: Path) -> List[Drone3DTargetPair]:
        """Trajectory-level batch processing"""
        trajectory_id = trajectory['path_id']
        scene_id = trajectory['scene_id']

        if self.verbose:
            print(f"Batch processing trajectory: {trajectory_id} ({len(trajectory['frame_pairs'])} pairs)")

        trajectory_landmarks = self.preprocess_trajectory_images(trajectory)

        nav_system = self.create_trajectory_navigation_system(trajectory_id)
        landmark_library = self.create_trajectory_landmark_library(trajectory_id, output_dir)

        all_landmarks_to_add = []
        image_to_landmarks_map = {}

        for image_path, landmarks in trajectory_landmarks.items():
            if landmarks:
                all_landmarks_to_add.extend(landmarks)
                image_to_landmarks_map[image_path] = landmarks

        landmark_id_map = {}
        for image_path, landmarks in image_to_landmarks_map.items():
            if landmarks:
                try:
                    merge_results = nav_system.topological_graph.add_image_landmarks(
                        image_path=image_path,
                        landmarks=landmarks,
                        timestamp=time.time()
                    )

                    if merge_results and 'merge_details' in merge_results:
                        for i, detail in enumerate(merge_results['merge_details']):
                            if 'target_id' in detail and i < len(landmarks):
                                key = f"{image_path}_{i}"
                                landmark_id_map[key] = detail['target_id']

                                landmark = landmarks[i]
                                landmark_info = Drone3DLandmarkInfo(
                                    landmark_id=detail['target_id'],
                                    trajectory_id=trajectory_id,
                                    image_path=image_path,
                                    quality_score=landmark.get('quality_score', 0),
                                    bbox=tuple(landmark.get('bbox', [0, 0, 100, 100])),
                                    center=tuple(landmark.get('center', [960, 540])),
                                    area=landmark.get('area', 0)
                                )
                                landmark_library.register_landmark(detail['target_id'], landmark_info)

                except Exception as e:
                    self.logger.error(f"Batch landmark addition failed: {image_path} - {e}")

        trajectory_target_pairs = []

        for pair in trajectory['frame_pairs']:
            try:
                target_pair = self.process_frame_pair_optimized(
                    trajectory, pair, trajectory_landmarks, landmark_id_map, landmark_library, nav_system
                )
                if target_pair:
                    trajectory_target_pairs.append(target_pair)
                    self.stats['valid_target_pairs'] += 1
                else:
                    self.stats['failed_pairs'] += 1

                self.stats['processed_pairs'] += 1

            except Exception as e:
                self.logger.error(f"Frame pair processing failed: {e}")
                self.stats['failed_pairs'] += 1

        if self.enable_visualization:
            self.visualize_trajectory_targets(trajectory_target_pairs, trajectory_id, output_dir)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.verbose:
            print(f"Trajectory {trajectory_id} processing complete: {len(trajectory_target_pairs)}/{len(trajectory['frame_pairs'])} valid pairs")

        return trajectory_target_pairs

    def process_frame_pair_optimized(self, trajectory: Dict, pair: Dict,
                                     trajectory_landmarks: Dict[str, List[Dict]],
                                     landmark_id_map: Dict[str, int],
                                     landmark_library: Drone3DLandmarkLibrary,
                                     nav_system) -> Optional[Drone3DTargetPair]:
        """Optimized frame pair processing (query from preprocessed cache)"""
        try:
            path_id = trajectory['path_id']
            scene_id = trajectory['scene_id']
            current_frame_id = pair['current_frame_id']
            navigation_frame_id = pair['navigation_frame_id']
            current_action = pair['current_action']
            frame_distance = pair.get('frame_distance', 0)
            pair_type = pair.get('pair_type', 'unknown')

            current_image_path = str(self.get_image_path(path_id, current_frame_id))
            navigation_image_path = str(self.get_image_path(path_id, navigation_frame_id))

            current_landmarks = trajectory_landmarks.get(current_image_path, [])
            navigation_landmarks = trajectory_landmarks.get(navigation_image_path, [])

            if not navigation_landmarks:
                self.stats['discarded_navigation_missing'] += 1
                return None

            navigation_target_id = None
            navigation_target_path = None
            navigation_target_confidence = 0.0

            if navigation_landmarks:
                landmark_idx, confidence = self.select_salient_landmark(navigation_landmarks)

                if landmark_idx is not None:
                    key = f"{navigation_image_path}_{landmark_idx}"
                    navigation_target_id = landmark_id_map.get(key)
                    if navigation_target_id is not None:
                        navigation_target_path = landmark_library.get_landmark_image_path(navigation_target_id)
                        navigation_target_confidence = confidence

            visible_target_id = None
            visible_target_path = None
            visible_target_confidence = 0.0

            if current_landmarks and navigation_target_id is not None:
                visible_target_id, visible_target_confidence = self._find_closest_visible_target(
                    current_landmarks, current_image_path, landmark_id_map, navigation_target_id, nav_system
                )
                if visible_target_id is not None:
                    visible_target_path = landmark_library.get_landmark_image_path(visible_target_id)
            elif current_landmarks:
                landmark_idx, confidence = self.select_salient_landmark(current_landmarks)
                if landmark_idx is not None:
                    key = f"{current_image_path}_{landmark_idx}"
                    visible_target_id = landmark_id_map.get(key)
                    if visible_target_id is not None:
                        visible_target_path = landmark_library.get_landmark_image_path(visible_target_id)
                        visible_target_confidence = confidence

            target_pair = Drone3DTargetPair(
                trajectory_id=path_id,
                scene_id=scene_id,
                current_frame_id=current_frame_id,
                navigation_frame_id=navigation_frame_id,
                current_action=current_action,
                visible_target_id=visible_target_id,
                visible_target_path=visible_target_path,
                visible_target_confidence=visible_target_confidence,
                navigation_target_id=navigation_target_id,
                navigation_target_path=navigation_target_path,
                navigation_target_confidence=navigation_target_confidence,
                current_landmarks_count=len(current_landmarks),
                navigation_landmarks_count=len(navigation_landmarks),
                frame_distance=frame_distance,
                pair_type=pair_type
            )

            self.stats['action_distribution'][current_action] += 1
            if visible_target_id is None:
                self.stats['no_landmarks_pairs'] += 1

            return target_pair

        except Exception as e:
            self.logger.error(f"Optimized frame pair processing failed: {e}")
            return None

    def process_dataset(self, max_trajectories: int = None,
                        output_file: str = "drone_3d_training_targets_optimized.json") -> List[Drone3DTargetPair]:
        """Process complete 3D drone dataset (optimized version)"""
        frame_pairs_data = self.load_frame_pairs()
        if not frame_pairs_data:
            return []

        trajectories = frame_pairs_data.get('trajectories', [])

        if max_trajectories:
            trajectories = trajectories[:max_trajectories]

        self.stats['total_trajectories'] = len(trajectories)
        self.stats['total_frame_pairs'] = sum(len(traj['frame_pairs']) for traj in trajectories)

        all_target_pairs = []

        output_dir = get_safe_output_dir()

        print(f"Processing {len(trajectories)} trajectories (optimized: model sharing + batch preprocessing + caching)...")

        with tqdm(total=len(trajectories), desc="Processing drone trajectories") as pbar:
            for trajectory in trajectories:
                try:
                    trajectory_pairs = self.process_trajectory_batch(trajectory, output_dir)
                    all_target_pairs.extend(trajectory_pairs)
                    pbar.update(1)

                except Exception as e:
                    self.logger.error(f"Trajectory processing failed: {trajectory.get('path_id', 'unknown')} - {e}")
                    pbar.update(1)
                    continue

        self.save_results(all_target_pairs, output_file, output_dir)
        self.print_statistics()

        return all_target_pairs

    def save_results(self, target_pairs: List[Drone3DTargetPair], output_file: str, output_dir: Path):
        """Save extracted target pairs (optimized version)"""
        try:
            output_path = output_dir / output_file

            serializable_pairs = []
            for pair in target_pairs:
                pair_dict = asdict(pair)
                if pair_dict['visible_target_path']:
                    pair_dict['visible_target_path'] = str(pair_dict['visible_target_path'])
                if pair_dict['navigation_target_path']:
                    pair_dict['navigation_target_path'] = str(pair_dict['navigation_target_path'])
                serializable_pairs.append(pair_dict)

            output_data = {
                'dataset': 'Drone3D',
                'processing_method': 'optimized_with_model_sharing_and_batch_processing',
                'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'action_space': self.action_names,
                'turn_actions': list(self.turn_actions),
                'optimization_features': {
                    'model_sharing': True,
                    'image_caching': True,
                    'batch_processing': True,
                    'trajectory_isolated_graphs': True,
                    'shared_model_instances': self.stats['shared_model_instances'],
                    'visualization_enabled': self.enable_visualization
                },
                'performance_stats': {
                    'cache_hits': self.stats['cache_hits'],
                    'cache_misses': self.stats['cache_misses'],
                    'unique_images_processed': self.stats['unique_images_processed'],
                    'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
                },
                'visible_target_strategy': 'topological_graph_based_closest_to_navigation_target',
                'navigation_target_strategy': 'saliency_based_unchanged',
                'statistics': dict(self.stats),
                'target_pairs': serializable_pairs,
                'total_pairs': len(serializable_pairs),
                'trajectory_landmark_libraries': {
                    traj_id: {
                        'base_directory': str(library.base_dir),
                        'total_landmarks': library.stats['total_landmarks']
                    }
                    for traj_id, library in self.trajectory_landmark_libraries.items()
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"Optimized results saved to: {output_path}")

            for trajectory_id, landmark_library in self.trajectory_landmark_libraries.items():
                try:
                    landmark_library.base_dir.mkdir(parents=True, exist_ok=True)

                    safe_trajectory_id = sanitize_trajectory_id(trajectory_id)
                    index_filename = f"trajectory_{safe_trajectory_id}_landmark_library_index.json"
                    index_path = landmark_library.base_dir / index_filename

                    index_data = {
                        'library_info': {
                            'trajectory_id': trajectory_id,
                            'safe_trajectory_id': safe_trajectory_id,
                            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'base_directory': str(landmark_library.base_dir),
                            'total_landmarks': landmark_library.stats['total_landmarks']
                        },
                        'landmarks': {
                            str(landmark_id): {
                                'landmark_id': info.landmark_id,
                                'trajectory_id': info.trajectory_id,
                                'image_path': info.image_path,
                                'quality_score': info.quality_score,
                                'bbox': info.bbox,
                                'center': info.center,
                                'area': info.area
                            }
                            for landmark_id, info in landmark_library.landmark_registry.items()
                        },
                        'statistics': landmark_library.stats.copy()
                    }

                    with open(index_path, 'w', encoding='utf-8') as f:
                        json.dump(index_data, f, indent=2, ensure_ascii=False)

                    if self.verbose:
                        print(f"Trajectory {trajectory_id} landmark library index saved: {index_path}")

                except Exception as e:
                    self.logger.error(f"Failed to save trajectory {trajectory_id} landmark library index: {e}")

        except Exception as e:
            print(f"Failed to save results: {e}")

    def print_statistics(self):
        """Print statistics (optimized version)"""
        print("\n" + "=" * 60)
        print("DRONE 3D TARGET EXTRACTION STATISTICS (OPTIMIZED)")
        print("=" * 60)
        print(f"Total trajectories: {self.stats['total_trajectories']}")
        print(f"Total frame pairs: {self.stats['total_frame_pairs']}")
        print(f"Processed pairs: {self.stats['processed_pairs']}")
        print(f"Valid target pairs: {self.stats['valid_target_pairs']}")
        print(f"No landmarks pairs: {self.stats['no_landmarks_pairs']}")
        print(f"Discarded (navigation missing): {self.stats['discarded_navigation_missing']}")
        print(f"Failed pairs: {self.stats['failed_pairs']}")

        if self.stats['processed_pairs'] > 0:
            success_rate = self.stats['valid_target_pairs'] / self.stats['processed_pairs'] * 100
            print(f"Success rate: {success_rate:.1f}%")

        print(f"\nOptimization effects:")
        print(f"- Shared model instances: {self.stats['shared_model_instances']}")
        print(f"- Model loading optimized: {'Enabled' if self.stats['model_loading_optimized'] else 'Disabled'}")
        print(f"- Cache hits: {self.stats['cache_hits']}")
        print(f"- Cache misses: {self.stats['cache_misses']}")
        print(f"- Cache hit rate: {self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']):.1%}")
        print(f"- Unique images processed: {self.stats['unique_images_processed']}")
        print(f"- Visualization: {'Enabled' if self.enable_visualization else 'Disabled'}")

        print(f"\nTarget extraction strategy:")
        print(f"- Visible target: Topological graph-based closest neighbor search (relative to navigation target)")
        print(f"- Navigation target: Saliency evaluation method (unchanged)")

        print(f"\n3D drone action distribution:")
        for action, count in sorted(self.stats['action_distribution'].items(), key=lambda x: str(x[0])):
            action_name = self.action_names.get(action, f'unknown_{action}')
            print(f"  {action_name}: {count} pairs")

        total_landmarks = sum(len(nav_sys.topological_graph.landmark_features)
                              for nav_sys in self.trajectory_nav_systems.values())
        print(f"\nTopological graph statistics:")
        print(f"  Independent trajectory graphs: {len(self.trajectory_nav_systems)}")
        print(f"  Total landmarks: {total_landmarks}")
        print(f"  Trajectory-specific libraries: {len(self.trajectory_landmark_libraries)}")
        print("=" * 60)

    def __del__(self):
        """Destructor: cleanup shared model instances"""
        try:
            if hasattr(self, 'shared_landmark_extractor') and self.shared_landmark_extractor:
                if self.verbose:
                    print("Cleaning up shared landmark extractor...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except:
            pass


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Drone 3D Target Extractor for Navigation Training (Optimized)')
    parser.add_argument('--drone_path', default='./drone_navigation_data',
                        help='Drone dataset base path')
    parser.add_argument('--frame_pairs', default='drone_3d_frame_pairs_full_dataset.json',
                        help='Frame pairs file path')
    parser.add_argument('--output', default='drone_3d_training_targets_optimized.json',
                        help='Output file name')
    parser.add_argument('--max_trajectories', type=int, default=None,
                        help='Maximum trajectories to process (for testing)')
    parser.add_argument('--model_size', type=str, default="7b", choices=['0.5b', '7b'],
                        help='LLaVA model size for scene analysis')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no_scene_analysis', action='store_true',
                        help='Disable scene analysis with LLaVA')
    parser.add_argument('--enable_visualization', action='store_true',
                        help='Enable visualization of trajectory targets (6 frame pairs per trajectory)')

    args = parser.parse_args()

    setup_environment()
    gpu_id, gpu_name, gpu_memory = get_best_gpu()

    fastsam_model_path = "FastSAM-x.pt"
    siglip_model_path = "google/siglip-so400m-patch14-384"
    dinov2_model_path = "/root/autodl-tmp/model/transformers/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415"

    llava_model_path = None
    if not args.no_scene_analysis:
        if args.model_size == "0.5b":
            llava_model_path = "/root/autodl-tmp/model/transformers/models--lmms-lab--llava-onevision-qwen2-0.5b-ov"
        else:
            llava_model_path = "/root/autodl-tmp/model/transformers/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"

    print("Starting optimized 3D drone target extractor...")
    print("Optimization features: model sharing + batch preprocessing + image caching + trajectory isolation")
    print("Target extraction strategy: visible target based on topological graph nearest neighbor + navigation target based on saliency")
    if args.enable_visualization:
        print("Visualization: Enabled (6 frame pairs per trajectory will be generated)")

    extractor = Drone3DTargetExtractor(
        drone_base_path=args.drone_path,
        frame_pairs_file=args.frame_pairs,
        fastsam_model_path=fastsam_model_path,
        siglip_model_path=siglip_model_path,
        dinov2_model_path=dinov2_model_path,
        llava_model_path=llava_model_path,
        preferred_gpu_id=gpu_id,
        verbose=args.verbose,
        enable_visualization=args.enable_visualization
    )

    print("Starting 3D drone target extraction (optimized version)...")
    start_time = time.time()

    target_pairs = extractor.process_dataset(
        max_trajectories=args.max_trajectories,
        output_file=args.output
    )

    elapsed_time = time.time() - start_time
    print(f"\nOptimized extraction complete, time: {elapsed_time:.1f}s")
    print(f"   Generated {len(target_pairs)} target pairs")
    print(f"   Performance improvement: model sharing + batch processing + caching mechanism")
    print(f"   Strategy innovation: visible target based on topological graph similarity search")
    if args.enable_visualization:
        print(f"   Visualization results saved in output directory's visualizations/ folder")


if __name__ == "__main__":
    main()
