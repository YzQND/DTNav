#!/usr/bin/env python3
"""
3D Drone Dataset Frame Pair Builder.
Constructs frame pairs for 3D drone navigation with extended action space.

Action space: None(start), move_forward, turn_left, turn_right, move_up, move_down, stop
"""

import json
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
import sys
from tqdm import tqdm


@dataclass
class Drone3DFramePair:
    """3D drone frame pair data structure"""
    current_frame_idx: int
    navigation_frame_idx: int
    current_frame_id: str
    navigation_frame_id: str
    current_action: str
    navigation_action: str
    frame_distance: int
    pair_type: str
    reasoning: str
    path_id: str
    scene_id: str


class Drone3DFramePairBuilder:
    """3D drone frame pair builder"""

    def __init__(self, verbose=False):
        self.action_names = {
            None: 'start',
            'move_forward': 'move_forward',
            'turn_left': 'turn_left',
            'turn_right': 'turn_right',
            'move_up': 'move_up',
            'move_down': 'move_down',
            'stop': 'stop'
        }

        self.turn_actions = {'turn_left', 'turn_right', 'move_up'}
        self.move_down_action = 'move_down'
        self.forward_action = 'move_forward'
        self.stop_action = 'stop'
        self.start_action = None

        self.stats = {
            'total_trajectories': 0,
            'total_frames': 0,
            'total_frame_pairs': 0,
            'stop_pairs': 0,
            'forward_pairs': 0,
            'turn_pairs': 0,
            'start_pairs': 0,
            'action_distribution': defaultdict(int),
            'frame_distance_distribution': defaultdict(int),
            'pair_type_distribution': defaultdict(int),
            'failed_trajectories': 0,
            'skipped_trajectories': 0,
            'move_down_pairs': 0,
            'final_frame_pairs': 0
        }

        self.verbose = verbose
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        handlers = [logging.FileHandler('drone_3d_processing.log', encoding='utf-8')]

        if self.verbose:
            handlers.append(logging.StreamHandler(sys.stdout))

        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def get_frame_action_truth(self, waypoints: List[Dict], frame_idx: int, total_frames: int) -> str:
        """Get action ground truth for specified frame"""
        if frame_idx >= total_frames - 1:
            return self.stop_action
        elif frame_idx + 1 < len(waypoints):
            next_waypoint = waypoints[frame_idx + 1]
            action = next_waypoint.get('action', self.stop_action)
            if action is None:
                return self.start_action
            return action
        else:
            return self.stop_action

    def find_turn_action_in_range(self, waypoints: List[Dict], start_idx: int, end_idx: int, total_frames: int) -> Optional[int]:
        """Find turn action position within specified range"""
        for frame_idx in range(start_idx, min(end_idx, total_frames)):
            action = self.get_frame_action_truth(waypoints, frame_idx, total_frames)
            if action in self.turn_actions:
                return frame_idx
        return None

    def build_frame_pairs_for_trajectory(self, waypoints: List[Dict], trajectory_id: str,
                                         scene_id: str = "") -> List[Drone3DFramePair]:
        """Build frame pairs for single 3D drone trajectory"""

        total_frames = len(waypoints)

        if total_frames < 2:
            return []

        frame_pairs = []

        for current_idx in range(total_frames):
            current_action = self.get_frame_action_truth(waypoints, current_idx, total_frames)
            current_waypoint = waypoints[current_idx]
            current_frame_id = current_waypoint.get('viewpoint_id', f"{trajectory_id}_{current_idx:04d}")

            self.stats['action_distribution'][current_action] += 1

            # Rule 1: STOP action pairs with itself
            if current_action == self.stop_action:
                pair = Drone3DFramePair(
                    current_frame_idx=current_idx,
                    navigation_frame_idx=current_idx,
                    current_frame_id=current_frame_id,
                    navigation_frame_id=current_frame_id,
                    current_action=current_action,
                    navigation_action=current_action,
                    frame_distance=0,
                    pair_type="stop_self",
                    reasoning="STOP action pairs with itself",
                    path_id=trajectory_id,
                    scene_id=scene_id
                )
                frame_pairs.append(pair)
                self.stats['stop_pairs'] += 1
                continue

            # Rule 2: START action pairs with itself
            elif current_action == self.start_action:
                pair = Drone3DFramePair(
                    current_frame_idx=current_idx,
                    navigation_frame_idx=current_idx,
                    current_frame_id=current_frame_id,
                    navigation_frame_id=current_frame_id,
                    current_action=current_action if current_action else "start",
                    navigation_action=current_action if current_action else "start",
                    frame_distance=0,
                    pair_type="start_self",
                    reasoning="START action pairs with itself",
                    path_id=trajectory_id,
                    scene_id=scene_id
                )
                frame_pairs.append(pair)
                self.stats['start_pairs'] += 1
                continue

            # Rule 3: FORWARD action
            elif current_action == self.forward_action:
                search_start = current_idx + 4
                search_end = current_idx + 9

                selected_nav_frames = []

                if self.verbose:
                    print(f"    Forward action at frame {current_idx}, search range {search_start}~{search_end - 1}")

                for nav_idx in range(search_start, min(search_end, total_frames)):
                    has_turn_between = False
                    turn_positions = []

                    for check_idx in range(current_idx + 1, nav_idx):
                        check_action = self.get_frame_action_truth(waypoints, check_idx, total_frames)
                        if check_action in self.turn_actions:
                            has_turn_between = True
                            turn_positions.append(check_idx)

                    nav_action = self.get_frame_action_truth(waypoints, nav_idx, total_frames)

                    if self.verbose:
                        print(f"      Frame {nav_idx}: action {nav_action}, turn positions {turn_positions}, selected: {has_turn_between}")

                    if has_turn_between:
                        selected_nav_frames.append(nav_idx)

                if not selected_nav_frames:
                    if self.verbose:
                        print(f"      No suitable frame in t+4~t+8, continue searching")

                    turn_found = False
                    for search_idx in range(search_end, total_frames):
                        search_action = self.get_frame_action_truth(waypoints, search_idx, total_frames)
                        if search_action in self.turn_actions:
                            nav_idx = search_idx + 1
                            if nav_idx < total_frames:
                                selected_nav_frames.append(nav_idx)
                                if self.verbose:
                                    print(f"      Found turn action at frame {search_idx}, select frame {nav_idx}")
                            turn_found = True
                            break

                    if not turn_found:
                        selected_nav_frames.append(total_frames - 1)
                        if self.verbose:
                            print(f"      No turn action found, select last frame {total_frames - 1}")

                for nav_idx in selected_nav_frames:
                    nav_waypoint = waypoints[nav_idx]
                    nav_frame_id = nav_waypoint.get('viewpoint_id', f"{trajectory_id}_{nav_idx:04d}")
                    nav_action = self.get_frame_action_truth(waypoints, nav_idx, total_frames)

                    pair = Drone3DFramePair(
                        current_frame_idx=current_idx,
                        navigation_frame_idx=nav_idx,
                        current_frame_id=current_frame_id,
                        navigation_frame_id=nav_frame_id,
                        current_action=current_action,
                        navigation_action=nav_action,
                        frame_distance=nav_idx - current_idx,
                        pair_type="forward_to_turn" if nav_action in self.turn_actions else "forward_to_end",
                        reasoning=f"Forward action, navigate to frame {nav_idx}",
                        path_id=trajectory_id,
                        scene_id=scene_id
                    )
                    frame_pairs.append(pair)
                    self.stats['forward_pairs'] += 1

            # Rule 4: MOVE_DOWN action
            elif current_action == self.move_down_action:
                search_start = current_idx + 1
                search_end = current_idx + 9
            
                for nav_idx in range(search_start, min(search_end, total_frames)):
                    nav_waypoint = waypoints[nav_idx]
                    nav_frame_id = nav_waypoint.get('viewpoint_id', f"{trajectory_id}_{nav_idx:04d}")
                    nav_action = self.get_frame_action_truth(waypoints, nav_idx, total_frames)
            
                    pair = Drone3DFramePair(
                        current_frame_idx=current_idx,
                        navigation_frame_idx=nav_idx,
                        current_frame_id=current_frame_id,
                        navigation_frame_id=nav_frame_id,
                        current_action=current_action,
                        navigation_action=nav_action,
                        frame_distance=nav_idx - current_idx,
                        pair_type="move_down_multi",
                        reasoning=f"Move down action, navigate to frame {nav_idx}",
                        path_id=trajectory_id,
                        scene_id=scene_id
                    )
                    frame_pairs.append(pair)
                    self.stats['move_down_pairs'] += 1
                        
            # Rule 5: Turn actions
            elif current_action in self.turn_actions:
                search_start = current_idx + 4
                search_end = current_idx + 9

                for nav_idx in range(search_start, min(search_end, total_frames)):
                    nav_waypoint = waypoints[nav_idx]
                    nav_frame_id = nav_waypoint.get('viewpoint_id', f"{trajectory_id}_{nav_idx:04d}")
                    nav_action = self.get_frame_action_truth(waypoints, nav_idx, total_frames)

                    pair = Drone3DFramePair(
                        current_frame_idx=current_idx,
                        navigation_frame_idx=nav_idx,
                        current_frame_id=current_frame_id,
                        navigation_frame_id=nav_frame_id,
                        current_action=current_action,
                        navigation_action=nav_action,
                        frame_distance=nav_idx - current_idx,
                        pair_type="turn_multi",
                        reasoning=f"Turn action ({current_action}), navigate to frame {nav_idx}",
                        path_id=trajectory_id,
                        scene_id=scene_id
                    )
                    frame_pairs.append(pair)
                    self.stats['turn_pairs'] += 1
        
            # Extra rule: Last 5 frames pair with final frame
            if current_idx >= total_frames - 5:
                final_frame_idx = total_frames - 1
                skip_final_pair = False
    
                if current_idx == final_frame_idx:
                    for existing_pair in frame_pairs:
                        if (existing_pair.current_frame_idx == current_idx and 
                            existing_pair.navigation_frame_idx == final_frame_idx and
                            existing_pair.pair_type == "stop_self"):
                            skip_final_pair = True
                            break
                
                if not skip_final_pair:
                    final_waypoint = waypoints[final_frame_idx]
                    final_frame_id = final_waypoint.get('viewpoint_id', f"{trajectory_id}_{final_frame_idx:04d}")
                
                    pair = Drone3DFramePair(
                        current_frame_idx=current_idx,
                        navigation_frame_idx=final_frame_idx,
                        current_frame_id=current_frame_id,
                        navigation_frame_id=final_frame_id,
                        current_action=current_action,
                        navigation_action=self.stop_action,
                        frame_distance=final_frame_idx - current_idx,
                        pair_type="final_frame_extra",
                        reasoning=f"Last 5 frames extra rule, navigate to final frame",
                        path_id=trajectory_id,
                        scene_id=scene_id
                    )
                    frame_pairs.append(pair)
                    self.stats['final_frame_pairs'] += 1

        for pair in frame_pairs:
            self.stats['frame_distance_distribution'][pair.frame_distance] += 1
            self.stats['pair_type_distribution'][pair.pair_type] += 1

        return frame_pairs

    def process_drone_trajectory(self, trajectory: Dict, trajectory_id: str) -> List[Drone3DFramePair]:
        """Process single 3D drone trajectory"""

        if 'path' not in trajectory:
            self.stats['skipped_trajectories'] += 1
            return []

        waypoints = trajectory['path']
        scene_id = trajectory.get('scene_id', '')

        if not waypoints:
            self.stats['skipped_trajectories'] += 1
            return []

        self.stats['total_trajectories'] += 1
        self.stats['total_frames'] += len(waypoints)

        try:
            frame_pairs = self.build_frame_pairs_for_trajectory(
                waypoints, trajectory_id, scene_id
            )
            self.stats['total_frame_pairs'] += len(frame_pairs)
            return frame_pairs

        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error processing trajectory {trajectory_id}: {e}")
            self.stats['failed_trajectories'] += 1
            return []

    def print_statistics(self, show_details: bool = True):
        """Print statistics"""
        if show_details:
            print(f"\n3D Drone Frame Pair Builder Statistics")
            print(f"Processed trajectories: {self.stats['total_trajectories']}")
            print(f"Successful trajectories: {self.stats['total_trajectories'] - self.stats['failed_trajectories']}")
            print(f"Failed trajectories: {self.stats['failed_trajectories']}")
            print(f"Skipped trajectories: {self.stats['skipped_trajectories']}")
            print(f"Total image frames: {self.stats['total_frames']}")
            print(f"Built frame pairs: {self.stats['total_frame_pairs']}")

            if self.stats['total_trajectories'] > 0:
                print(f"Avg frames per trajectory: {self.stats['total_frames'] / self.stats['total_trajectories']:.1f}")
                print(f"Avg pairs per trajectory: {self.stats['total_frame_pairs'] / self.stats['total_trajectories']:.1f}")

            print(f"\nFrame pair type distribution:")
            print(f"  Start pairs: {self.stats['start_pairs']}")
            print(f"  STOP pairs: {self.stats['stop_pairs']}")
            print(f"  Forward pairs: {self.stats['forward_pairs']}")
            print(f"  Turn pairs: {self.stats['turn_pairs']}")
            print(f"  Move down pairs: {self.stats['move_down_pairs']}")
            print(f"  Final frame pairs: {self.stats['final_frame_pairs']}")

            print(f"\nAction distribution:")
            for action in sorted(self.stats['action_distribution'].keys(), key=lambda x: str(x)):
                count = self.stats['action_distribution'][action]
                action_name = self.action_names.get(action, f'unknown_{action}')
                print(f"  {action_name}: {count}")

            print(f"\nFrame distance distribution (top 10):")
            sorted_distances = sorted(self.stats['frame_distance_distribution'].items(),
                                      key=lambda x: x[1], reverse=True)
            for distance, count in sorted_distances[:10]:
                print(f"  Distance {distance}: {count} pairs")

        else:
            print(f"Completed: {self.stats['total_trajectories']} trajectories -> {self.stats['total_frame_pairs']} pairs")

    def save_frame_pairs(self, frame_pairs: List[Drone3DFramePair], output_path: str, silent: bool = False):
        """Save frame pairs to JSON file"""
        trajectory_groups = defaultdict(list)

        for pair in frame_pairs:
            trajectory_groups[pair.path_id].append(pair)

        trajectories_data = []

        for path_id, pairs in trajectory_groups.items():
            first_pair = pairs[0]
            scene_id = first_pair.scene_id

            trajectory_pairs = []
            for pair in pairs:
                trajectory_pairs.append({
                    'current_frame_id': pair.current_frame_id,
                    'navigation_frame_id': pair.navigation_frame_id,
                    'current_action': pair.current_action,
                    'frame_distance': pair.frame_distance,
                    'pair_type': pair.pair_type
                })

            trajectory_data = {
                'path_id': path_id,
                'scene_id': scene_id,
                'frame_pairs': trajectory_pairs,
                'pair_count': len(trajectory_pairs)
            }

            trajectories_data.append(trajectory_data)

        trajectories_data.sort(key=lambda x: x['path_id'])

        output_data = {
            'dataset': 'Drone3D',
            'action_space': {
                'None': 'start',
                'move_forward': 'move_forward',
                'turn_left': 'turn_left',
                'turn_right': 'turn_right',
                'move_up': 'move_up',
                'move_down': 'move_down',
                'stop': 'stop'
            },
            'turn_actions': list(self.turn_actions),
            'trajectories': trajectories_data,
            'total_trajectories': len(trajectories_data),
            'total_pairs': len(frame_pairs),
            'processing_stats': {
                'processed_trajectories': self.stats['total_trajectories'],
                'failed_trajectories': self.stats['failed_trajectories'],
                'skipped_trajectories': self.stats['skipped_trajectories'],
                'success_rate': ((self.stats['total_trajectories'] - self.stats['failed_trajectories']) /
                                 max(self.stats['total_trajectories'], 1)) * 100
            },
            'pair_statistics': {
                'start_pairs': self.stats['start_pairs'],
                'stop_pairs': self.stats['stop_pairs'],
                'forward_pairs': self.stats['forward_pairs'],
                'turn_pairs': self.stats['turn_pairs'],
                'move_down_pairs': self.stats['move_down_pairs'],
                'final_frame_pairs': self.stats['final_frame_pairs'],
                'avg_pairs_per_trajectory': len(frame_pairs) / len(trajectories_data) if trajectories_data else 0
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        if not silent:
            print(f"Results saved to: {output_path}")


def process_full_drone_dataset(annotation_file: str,
                               output_file: str = None,
                               verbose: bool = False):
    """Process full 3D drone dataset"""

    if output_file is None:
        output_file = "drone_3d_frame_pairs_full_dataset.json"

    if not Path(annotation_file).exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'trajectories' in data:
            trajectories = data['trajectories']
        else:
            trajectories = data

        total_trajectories = len(trajectories)

        if total_trajectories == 0:
            print("No trajectory data found")
            return None, []

        builder = Drone3DFramePairBuilder(verbose=verbose)
        all_frame_pairs = []

        with tqdm(total=total_trajectories, desc="Processing 3D drone trajectories",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            for i, trajectory in enumerate(trajectories):
                trajectory_id = str(trajectory.get('path_id', f'trajectory_{i}'))

                try:
                    frame_pairs = builder.process_drone_trajectory(trajectory, trajectory_id)
                    all_frame_pairs.extend(frame_pairs)

                except Exception as e:
                    if verbose:
                        print(f"\nError processing trajectory {i + 1}: {e}")
                    builder.stats['failed_trajectories'] += 1

                pbar.update(1)

        builder.save_frame_pairs(all_frame_pairs, output_file, silent=not verbose)
        builder.print_statistics(show_details=verbose)

        return builder, all_frame_pairs

    except Exception as e:
        print(f"Processing failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None, []


def test_drone_frame_pair_builder():
    """Test 3D drone frame pair builder"""
    annotation_file = "./drone_navigation_data/annotations.json"

    if not Path(annotation_file).exists():
        print(f"Annotation file not found: {annotation_file}")
        print("Please generate data using drone_collector_r2r.py first")
        return None, []

    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'trajectories' in data:
            trajectories = data['trajectories']
        else:
            trajectories = [data] if isinstance(data, dict) else data

        builder = Drone3DFramePairBuilder(verbose=True)

        test_count = min(3, len(trajectories))
        all_frame_pairs = []

        print(f"Testing first {test_count} 3D drone trajectories...")

        for i, trajectory in enumerate(trajectories[:test_count]):
            trajectory_id = str(trajectory.get('path_id', f'trajectory_{i}'))

            print(f"\n--- Processing trajectory {i + 1}: ID={trajectory_id} ---")
            if 'path' in trajectory:
                waypoints = trajectory['path']
                print(f"Waypoint count: {len(waypoints)}")
                if waypoints:
                    print(f"First action: {waypoints[0].get('action', 'None')}")
                    print(f"Last action: {waypoints[-1].get('action', 'None')}")
                    actions = [wp.get('action') for wp in waypoints]
                    print(f"Action sequence: {actions[:10]}{'...' if len(actions) > 10 else ''}")

            frame_pairs = builder.process_drone_trajectory(trajectory, trajectory_id)
            all_frame_pairs.extend(frame_pairs)

            print(f"Generated {len(frame_pairs)} frame pairs")

        builder.print_statistics()

        output_path = "drone_3d_frame_pairs_test.json"
        builder.save_frame_pairs(all_frame_pairs, output_path)

        return builder, all_frame_pairs

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='3D Drone Dataset Frame Pair Builder')
    parser.add_argument('--mode', choices=['test', 'full'], default='full',
                        help='Run mode: test=test first few trajectories, full=process complete dataset')
    parser.add_argument('--input', default='./drone_navigation_data/annotations.json',
                        help='Input annotation file path')
    parser.add_argument('--output', default='drone_3d_frame_pairs_full_dataset.json',
                        help='Output file path')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose mode for detailed output')

    args = parser.parse_args()

    if args.mode == 'test':
        builder, frame_pairs = test_drone_frame_pair_builder()
        if builder and frame_pairs:
            print(f"\nTest completed! Built {len(frame_pairs)} frame pairs")

    elif args.mode == 'full':
        start_time = time.time()

        builder, frame_pairs = process_full_drone_dataset(
            annotation_file=args.input,
            output_file=args.output,
            verbose=args.verbose
        )

        total_time = time.time() - start_time

        if builder and frame_pairs:
            print(f"\nProcessing completed! Time: {total_time:.1f}s, Total: {len(frame_pairs)} pairs")
        else:
            print("Processing failed")
