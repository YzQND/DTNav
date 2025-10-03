#!/usr/bin/env python3
"""
Landmark extraction module using SigLIP for feature alignment.
Supports RGB landmark image storage for multi-image navigation.
Uses FastSAM for improved speed and batch processing optimization.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import warnings
import os
import time
from datetime import datetime

warnings.filterwarnings('ignore')

from utils import resize_image_if_needed, get_best_gpu, set_gpu_device, get_safe_output_dir
from ultralytics import FastSAM
from transformers import AutoProcessor, AutoModel
from transformers import AutoImageProcessor, AutoModel as DINOv2Model


class SigLIPEncoder:
    """SigLIP encoder for vision-text feature alignment and similarity computation."""

    def __init__(self, siglip_model_path="google/siglip-so400m-patch14-384",
                 preferred_gpu_id=None, use_fp16=True):
        """
        Initialize SigLIP encoder.
        
        Args:
            siglip_model_path: Path to SigLIP model
            preferred_gpu_id: Preferred GPU ID
            use_fp16: Use FP16 precision (default True)
        """
        if preferred_gpu_id is None:
            gpu_id, _, _ = get_best_gpu()
        else:
            gpu_id = preferred_gpu_id

        self.device = set_gpu_device(gpu_id)
        self.use_fp16 = use_fp16
        self._load_siglip_model(siglip_model_path)

        # Optimized parameters
        self.optimal_temperature = 0.07
        self.learnable_bias = 0.0
        self.similarity_strategy = "weighted_ensemble"
        self.normalization_strategy = "sigmoid_scaling"

        # Query template cache
        self.query_templates_cache = {}
        self.text_features_cache = {}

    def _load_siglip_model(self, model_path):
        """Load SigLIP model with configurable precision."""
        dtype = torch.float16 if self.use_fp16 else torch.float32

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": self.device},
            local_files_only=False,
            trust_remote_code=True
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=False,
            trust_remote_code=True
        )

        if self.use_fp16:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

    def _create_optimized_query_templates(self, query_text: str) -> List[str]:
        """Create optimized query templates."""
        if query_text in self.query_templates_cache:
            return self.query_templates_cache[query_text]

        templates = [
            f"photo of {query_text}",
            f"image of {query_text}",
            f"a {query_text}",
            f"{query_text} object",
            f"{query_text}",
        ]

        self.query_templates_cache[query_text] = templates
        return templates

    def compute_siglip_similarity_with_bias(self, text_features, image_features, temperature=None, bias=None):
        """Compute SigLIP similarity with bias."""
        try:
            if temperature is None:
                temperature = self.optimal_temperature
            if bias is None:
                bias = self.learnable_bias

            text_feat = F.normalize(text_features, p=2, dim=1)
            image_feat = F.normalize(image_features, p=2, dim=1)

            dot_product = torch.matmul(image_feat, text_feat.T)
            logits = (dot_product / temperature) + bias
            similarity = torch.sigmoid(logits).squeeze(-1)

            return float(similarity.item())
        except Exception:
            return 0.0

    def compute_enhanced_similarity(self, text_features, image_features):
        """Enhanced similarity computation with multi-temperature ensemble."""
        base_similarity = self.compute_siglip_similarity_with_bias(
            text_features, image_features,
            self.optimal_temperature, self.learnable_bias
        )

        if self.similarity_strategy == "weighted_ensemble":
            temp_weights = {0.01: 0.1, 0.05: 0.3, 0.07: 0.4, 0.1: 0.15, 0.2: 0.05}
            weighted_sum = 0.0
            total_weight = 0.0

            for temp, weight in temp_weights.items():
                sim = self.compute_siglip_similarity_with_bias(
                    text_features, image_features, temp, self.learnable_bias
                )
                weighted_sum += sim * weight
                total_weight += weight

            ensemble_similarity = weighted_sum / total_weight if total_weight > 0 else base_similarity
            return ensemble_similarity

        return base_similarity

    def encode_image_batch(self, pil_images):
        """Batch encode images."""
        if self.model is None or self.processor is None:
            return None

        try:
            if not pil_images:
                return None

            rgb_images = []
            for pil_image in pil_images:
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                rgb_images.append(pil_image)

            image_inputs = self.processor(images=rgb_images, return_tensors="pt")
            model_dtype = next(self.model.parameters()).dtype
            pixel_values = image_inputs['pixel_values'].to(self.device, dtype=model_dtype)

            with torch.no_grad():
                model_inputs = {'pixel_values': pixel_values}
                image_features = self.model.get_image_features(**model_inputs)
                image_features = F.normalize(image_features, p=2, dim=1)
                return image_features.cpu().float()
        except Exception:
            return None

    def encode_image(self, pil_image):
        """Extract visual features using SigLIP."""
        if self.model is None or self.processor is None:
            return None

        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            image_inputs = self.processor(images=pil_image, return_tensors="pt")
            model_dtype = next(self.model.parameters()).dtype
            pixel_values = image_inputs['pixel_values'].to(self.device, dtype=model_dtype)

            with torch.no_grad():
                model_inputs = {'pixel_values': pixel_values}
                image_features = self.model.get_image_features(**model_inputs)
                image_features = F.normalize(image_features, p=2, dim=1)
                return image_features.cpu().float()
        except Exception:
            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    return self.encode_image(pil_image)
            except:
                return None

    def encode_text(self, text_instruction):
        """Extract text features using SigLIP."""
        if self.model is None or self.processor is None:
            return None

        try:
            if isinstance(text_instruction, str):
                text_instruction = [text_instruction]

            max_length = getattr(self.processor.tokenizer, 'model_max_length', 77)
            if hasattr(self.model.config, 'text_config'):
                max_length = getattr(self.model.config.text_config, 'max_position_embeddings', 64)

            text_inputs = self.processor(
                text=text_instruction,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=min(max_length, 64)
            ).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
                return text_features.cpu().float()
        except Exception:
            return None

    def compute_similarity(self, text_instruction, visual_features_list):
        """Compute vision-text similarity using SigLIP."""
        templates = self._create_optimized_query_templates(text_instruction)

        template_features = []
        for template in templates:
            if template in self.text_features_cache:
                features = self.text_features_cache[template]
            else:
                features = self.encode_text(template)
                if features is not None:
                    self.text_features_cache[template] = features

            if features is not None:
                template_features.append(features)

        if not template_features:
            return []

        similarities = []

        for landmark_id, visual_features in visual_features_list:
            if visual_features is None:
                continue

            model_dtype = next(self.model.parameters()).dtype
            visual_features = visual_features.to(device=self.device, dtype=model_dtype)
            visual_features = F.normalize(visual_features, p=2, dim=1)

            template_similarities = []

            for text_features in template_features:
                text_features_device = text_features.to(device=self.device, dtype=model_dtype)
                text_features_device = F.normalize(text_features_device, p=2, dim=1)

                similarity = self.compute_enhanced_similarity(text_features_device, visual_features)
                template_similarities.append(similarity)

            if template_similarities:
                if self.similarity_strategy == "weighted_ensemble":
                    max_sim = max(template_similarities)
                    avg_sim = np.mean(template_similarities)
                    final_score = max_sim * 0.4 + avg_sim * 0.6
                elif self.similarity_strategy == "max_pooling":
                    final_score = max(template_similarities)
                else:
                    final_score = np.mean(template_similarities)

                if self.normalization_strategy == "sigmoid_scaling":
                    final_score = torch.sigmoid(torch.tensor(final_score * 10 - 5)).item()

                if not np.isfinite(final_score):
                    final_score = 0.0

                similarities.append((landmark_id, final_score))

        return similarities


class LandmarkExtractor:
    """Landmark extractor using FastSAM for feature alignment (supports RGB image storage)."""

    def __init__(self, fastsam_model_path="FastSAM-s.pt",
                 siglip_model_path="google/siglip-so400m-patch14-384",
                 dinov2_model=None,
                 preferred_gpu_id=None,
                 use_fp16=True,
                 landmark_padding_ratio=0.1):

        if preferred_gpu_id is None:
            gpu_id, _, _ = get_best_gpu()
        else:
            gpu_id = preferred_gpu_id

        self.device = set_gpu_device(gpu_id)

        self._initialize_fastsam(fastsam_model_path)

        self.siglip_encoder = None
        if siglip_model_path:
            try:
                self.siglip_encoder = SigLIPEncoder(
                    siglip_model_path,
                    preferred_gpu_id=gpu_id,
                    use_fp16=use_fp16
                )
            except Exception:
                pass

        self._initialize_dinov2(dinov2_model)

        # Quality and area thresholds
        self.quality_threshold = 0.30
        self.min_area_threshold = 1500
        self.max_area_threshold = 100000
        self.max_landmarks_per_image = 5
        self.edge_margin_threshold = 10

        # FastSAM configuration
        self.fastsam_conf_threshold = 0.4
        self.fastsam_iou_threshold = 0.9
        self.fastsam_imgsz = 1024

        # Batch processing configuration
        self.max_batch_size = 8
        self.target_size = (224, 224)

        self._setup_landmark_image_storage()

        # Landmark extraction configuration
        self.landmark_padding_ratio = landmark_padding_ratio
        self.use_rectangular_regions = True
        self.min_padding_pixels = 5
        self.max_padding_pixels = 50

    def _setup_landmark_image_storage(self):
        """Setup landmark image storage directory."""
        try:
            base_output_dir = get_safe_output_dir()
            self.landmark_images_dir = base_output_dir / "landmark_regions"
            self.landmark_images_dir.mkdir(parents=True, exist_ok=True)

            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.landmark_images_dir / f"session_{session_timestamp}"
            self.session_dir.mkdir(exist_ok=True)
        except Exception:
            self.landmark_images_dir = Path("/tmp/landmark_regions")
            self.landmark_images_dir.mkdir(exist_ok=True)
            self.session_dir = self.landmark_images_dir

    def _initialize_fastsam(self, model_path):
        """Initialize FastSAM model."""
        try:
            self.fastsam_model = FastSAM(model_path)
        except Exception:
            self.fastsam_model = None

    def _initialize_dinov2(self, dinov2_model):
        """Initialize DINOv2 model."""
        try:
            self.dinov2_processor = AutoImageProcessor.from_pretrained(
                dinov2_model, local_files_only=True
            )
            self.dinov2_model = DINOv2Model.from_pretrained(
                dinov2_model, local_files_only=True
            ).to(self.device)
            self.dinov2_model.eval()
        except Exception:
            self.dinov2_model = None

    def _prepare_batch_images(self, region_images):
        """Batch preprocess images to uniform format."""
        processed_images = []
        valid_indices = []

        for i, region_image in enumerate(region_images):
            if region_image is None or region_image.size == 0:
                continue

            try:
                if region_image.shape[0] < 32 or region_image.shape[1] < 32:
                    region_image = cv2.resize(region_image, (32, 32))

                pil_image = Image.fromarray(region_image.astype(np.uint8))
                processed_images.append(pil_image)
                valid_indices.append(i)
            except Exception:
                continue

        return processed_images, valid_indices

    def _prepare_dinov2_batch(self, region_images):
        """Prepare batch input for DINOv2."""
        processed_images = []
        valid_indices = []

        for i, region_image in enumerate(region_images):
            if region_image is None or region_image.size == 0:
                continue

            try:
                if region_image.shape[0] < 32 or region_image.shape[1] < 32:
                    region_image = cv2.resize(region_image, (32, 32))

                pil_image = Image.fromarray(region_image.astype(np.uint8))
                processed_images.append(pil_image)
                valid_indices.append(i)
            except Exception:
                continue

        return processed_images, valid_indices

    def _extract_siglip_features_batch(self, region_images):
        """Batch extract SigLIP features."""
        if not self.siglip_encoder or not self.siglip_encoder.model:
            return [None] * len(region_images)

        try:
            processed_images, valid_indices = self._prepare_batch_images(region_images)

            if not processed_images:
                return [None] * len(region_images)

            batch_size = min(self.max_batch_size, len(processed_images))
            all_features = []

            for i in range(0, len(processed_images), batch_size):
                batch_images = processed_images[i:i + batch_size]
                batch_features = self.siglip_encoder.encode_image_batch(batch_images)

                if batch_features is not None:
                    for j in range(batch_features.shape[0]):
                        all_features.append(batch_features[j:j + 1])
                else:
                    for img in batch_images:
                        feature = self.siglip_encoder.encode_image(img)
                        all_features.append(feature)

            features_result = [None] * len(region_images)
            for i, original_idx in enumerate(valid_indices):
                if i < len(all_features):
                    features_result[original_idx] = all_features[i]

            return features_result
        except Exception:
            return [self._extract_siglip_features(img) for img in region_images]

    def _extract_dinov2_features_batch(self, region_images):
        """Batch extract DINOv2 features."""
        if self.dinov2_model is None:
            return [None] * len(region_images)

        try:
            processed_images, valid_indices = self._prepare_dinov2_batch(region_images)

            if not processed_images:
                return [None] * len(region_images)

            batch_size = min(self.max_batch_size, len(processed_images))
            all_features = []

            for i in range(0, len(processed_images), batch_size):
                batch_images = processed_images[i:i + batch_size]

                try:
                    inputs = self.dinov2_processor(images=batch_images, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = self.dinov2_model(**inputs)
                        features = outputs.last_hidden_state[:, 0]
                        features = F.normalize(features, p=2, dim=1)

                        for j in range(features.shape[0]):
                            all_features.append(features[j:j + 1].cpu().float())
                except Exception:
                    for img in batch_images:
                        feature = self._extract_dinov2_features(img)
                        all_features.append(feature)

            features_result = [None] * len(region_images)
            for i, original_idx in enumerate(valid_indices):
                if i < len(all_features):
                    features_result[original_idx] = all_features[i]

            return features_result
        except Exception:
            return [self._extract_dinov2_features(img) for img in region_images]

    def extract_landmarks(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """
        Extract landmarks from image.
        
        Returns:
            Tuple of (landmarks list, timing info dict)
        """
        timing_info = {
            'fastsam_segmentation_time': 0.0,
            'siglip_feature_time': 0.0,
            'dinov2_feature_time': 0.0,
            'quality_computation_time': 0.0,
            'filtering_time': 0.0,
            'image_saving_time': 0.0,
            'total_time': 0.0
        }

        total_start = time.time()

        if not Path(image_path).exists():
            timing_info['total_time'] = time.time() - total_start
            return [], timing_info

        image = cv2.imread(image_path)
        if image is None:
            timing_info['total_time'] = time.time() - total_start
            return [], timing_info

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb, scale_factor = resize_image_if_needed(image_rgb, min_size=512)

        # FastSAM segmentation
        fastsam_start = time.time()
        segments = self._extract_fastsam_segments(image_rgb)
        segments = self._filter_edge_segments(segments, image_rgb.shape)
        timing_info['fastsam_segmentation_time'] = time.time() - fastsam_start

        if not segments:
            timing_info['total_time'] = time.time() - total_start
            return [], timing_info

        # Extract region images
        region_images = []
        valid_segments = []

        for segment in segments:
            region_image = self._extract_region_image(image_rgb, segment['mask'], segment['bbox'])
            if region_image is not None and region_image.size > 0:
                region_images.append(region_image)
                valid_segments.append(segment)

        if not region_images:
            timing_info['total_time'] = time.time() - total_start
            return [], timing_info

        # Batch feature extraction
        siglip_start = time.time()
        siglip_features_batch = self._extract_siglip_features_batch(region_images)
        timing_info['siglip_feature_time'] = time.time() - siglip_start

        dinov2_start = time.time()
        dinov2_features_batch = self._extract_dinov2_features_batch(region_images)
        timing_info['dinov2_feature_time'] = time.time() - dinov2_start

        # Quality computation and filtering
        quality_start = time.time()
        candidate_landmarks = []

        for i, (segment, region_image) in enumerate(zip(valid_segments, region_images)):
            features = {}
            if i < len(siglip_features_batch) and siglip_features_batch[i] is not None:
                features['siglip'] = siglip_features_batch[i]
            if i < len(dinov2_features_batch) and dinov2_features_batch[i] is not None:
                features['dinov2'] = dinov2_features_batch[i]

            quality_score = self._compute_quality_score(segment, region_image, features)

            center_y, center_x = np.where(segment['mask'])
            if len(center_x) == 0:
                continue

            landmark = {
                'id': segment['id'],
                'mask': segment['mask'],
                'bbox': [int(x / scale_factor) for x in segment['bbox']],
                'center': (int(np.mean(center_x) / scale_factor), int(np.mean(center_y) / scale_factor)),
                'area': segment['area'] / (scale_factor ** 2),
                'stability_score': segment['stability_score'],
                'predicted_iou': segment.get('predicted_iou', 1.0),
                'confidence': segment.get('confidence', 1.0),
                'quality_score': quality_score,
                'features': features,
                'image_path': image_path,
                'region_image': region_image,
                'scale_factor': scale_factor,
            }

            if self._passes_quality_check(landmark):
                candidate_landmarks.append(landmark)

        timing_info['quality_computation_time'] = time.time() - quality_start

        # Filtering
        filtering_start = time.time()
        filtered_landmarks = self._filter_landmarks(candidate_landmarks)
        timing_info['filtering_time'] = time.time() - filtering_start

        # Save landmark images
        saving_start = time.time()
        final_landmarks = []
        for landmark in filtered_landmarks:
            landmark_image_path = self._save_landmark_region_image(
                landmark, image_rgb, image_path
            )
            landmark['landmark_image_path'] = landmark_image_path
            final_landmarks.append(landmark)
        timing_info['image_saving_time'] = time.time() - saving_start

        timing_info['total_time'] = time.time() - total_start

        return final_landmarks, timing_info

    def _is_edge_segment(self, segment, image_shape):
        """Check if segment is too close to image edges."""
        bbox = segment['bbox']
        x, y, w, h = bbox
        img_height, img_width = image_shape[:2]

        left_edge = x
        top_edge = y
        right_edge = x + w
        bottom_edge = y + h

        near_left = left_edge <= self.edge_margin_threshold
        near_top = top_edge <= self.edge_margin_threshold
        near_right = right_edge >= (img_width - self.edge_margin_threshold)
        near_bottom = bottom_edge >= (img_height - self.edge_margin_threshold)

        return near_left or near_top or near_right or near_bottom

    def _filter_edge_segments(self, segments, image_shape):
        """Filter edge segments."""
        return [segment for segment in segments if not self._is_edge_segment(segment, image_shape)]

    def _extract_fastsam_segments(self, image_rgb):
        """Extract FastSAM segments."""
        if self.fastsam_model is None:
            return []

        try:
            results = self.fastsam_model(image_rgb,
                                         device=self.device,
                                         retina_masks=True,
                                         imgsz=self.fastsam_imgsz,
                                         conf=self.fastsam_conf_threshold,
                                         iou=self.fastsam_iou_threshold,
                                         verbose=False)

            segments = []

            for i, result in enumerate(results):
                if result.masks is None:
                    continue

                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for j, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                    mask_bool = mask > 0.5

                    x1, y1, x2, y2 = box
                    bbox_format = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

                    area = np.sum(mask_bool)

                    segment = {
                        'id': len(segments),
                        'mask': mask_bool,
                        'bbox': bbox_format,
                        'area': area,
                        'stability_score': float(score),
                        'predicted_iou': float(score),
                        'confidence': float(score),
                    }
                    segments.append(segment)

            return segments
        except Exception:
            return []

    def _save_landmark_region_image(self, landmark, image_rgb, image_path):
        """Save rectangular landmark region image with padding."""
        try:
            bbox = landmark['bbox']
            scale_factor = landmark['scale_factor']

            scaled_bbox = [int(x * scale_factor) for x in bbox]
            x, y, w, h = scaled_bbox

            # Compute padding
            padding_w = max(self.min_padding_pixels,
                            min(self.max_padding_pixels, int(w * self.landmark_padding_ratio)))
            padding_h = max(self.min_padding_pixels,
                            min(self.max_padding_pixels, int(h * self.landmark_padding_ratio)))

            # Apply padding within image boundaries
            img_height, img_width = image_rgb.shape[:2]

            padded_x = max(0, x - padding_w)
            padded_y = max(0, y - padding_h)
            padded_x2 = min(img_width, x + w + padding_w)
            padded_y2 = min(img_height, y + h + padding_h)

            actual_w = padded_x2 - padded_x
            actual_h = padded_y2 - padded_y

            if actual_w <= 0 or actual_h <= 0:
                return None

            # Extract rectangular region
            if self.use_rectangular_regions:
                landmark_image = image_rgb[padded_y:padded_y2, padded_x:padded_x2].copy()
            else:
                mask = landmark['mask']
                landmark_image = self._extract_mask_based_region(
                    image_rgb, mask, padded_x, padded_y, actual_w, actual_h
                )

            if landmark_image is None or landmark_image.size == 0:
                return None

            # Generate filename
            original_name = Path(image_path).stem
            timestamp = int(time.time() * 1000) % 100000

            padding_info = f"p{padding_w}x{padding_h}" if self.use_rectangular_regions else "mask"
            landmark_filename = f"{original_name}_L{landmark['id']}_{padding_info}_{timestamp}.png"
            landmark_path = self.session_dir / landmark_filename

            # Save as PNG
            landmark_bgr = cv2.cvtColor(landmark_image, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(landmark_path), landmark_bgr)

            if success:
                landmark['padded_bbox'] = [padded_x, padded_y, actual_w, actual_h]
                landmark['padding_applied'] = [padding_w, padding_h]
                return str(landmark_path)
            else:
                return None
        except Exception:
            return None

    def _extract_mask_based_region(self, image_rgb, mask, x, y, w, h):
        """Alternative: mask-based region extraction."""
        try:
            region = image_rgb[y:y + h, x:x + w].copy()
            mask_crop = mask[y:y + h, x:x + w] if mask.shape == image_rgb.shape[:2] else mask

            if len(region.shape) == 3 and mask_crop.shape == region.shape[:2]:
                mask_3d = np.stack([mask_crop, mask_crop, mask_crop], axis=2)
                region = np.where(mask_3d, region, 255)

            return region
        except:
            return None

    def configure_landmark_extraction(self, padding_ratio=0.1, use_rectangular=True,
                                      normalize_size=False, mark_bbox=False):
        """Configure landmark extraction parameters."""
        self.landmark_padding_ratio = padding_ratio
        self.use_rectangular_regions = use_rectangular
        self.normalize_landmark_size = normalize_size

    def _extract_siglip_features(self, region_image):
        """Extract SigLIP features."""
        if not self.siglip_encoder or not self.siglip_encoder.model:
            return None

        try:
            if region_image.shape[0] < 32 or region_image.shape[1] < 32:
                region_image = cv2.resize(region_image, (32, 32))

            pil_image = Image.fromarray(region_image.astype(np.uint8))
            features = self.siglip_encoder.encode_image(pil_image)
            return features
        except Exception:
            return None

    def _extract_dinov2_features(self, region_image):
        """Extract DINOv2 features."""
        if self.dinov2_model is None:
            return None

        try:
            if region_image.shape[0] < 32 or region_image.shape[1] < 32:
                region_image = cv2.resize(region_image, (32, 32))

            pil_image = Image.fromarray(region_image.astype(np.uint8))
            inputs = self.dinov2_processor(images=pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.dinov2_model(**inputs)
                features = outputs.last_hidden_state[:, 0]

                if features.dim() > 1 and features.shape[0] > 1:
                    features = features.mean(dim=0, keepdim=True)
                if features.dim() == 1:
                    features = features.unsqueeze(0)

                features = F.normalize(features, p=2, dim=1)

            return features.cpu().float()
        except Exception:
            return None

    def _extract_region_image(self, image, mask, bbox):
        """Extract region image."""
        try:
            x, y, w, h = [int(coord) for coord in bbox]
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            if w <= 0 or h <= 0:
                return None

            region = image[y:y + h, x:x + w].copy()
            region_mask = mask[y:y + h, x:x + w]

            if region_mask.any():
                if len(region.shape) == 3:
                    mask_3d = np.stack([region_mask, region_mask, region_mask], axis=2)
                    region = region * mask_3d

            return region
        except Exception:
            return None

    def _compute_quality_score(self, segment, region_image, features):
        """Compute quality score."""
        score = 0.0

        # Confidence score
        confidence = segment.get('confidence', 0.5)
        score += 0.50 * confidence

        # Stability score
        score += 0.20 * segment.get('stability_score', 0.5)

        # Area score
        area = segment['area']
        if area < 3000:
            area_score = area / 5000.0
        elif area < 15000:
            area_score = 0.8 + (area - 3000) / 60000.0
        elif area < 50000:
            area_score = 1.0
        else:
            area_score = max(0.8, 1.0 - (area - 50000) / 100000)
        score += 0.25 * area_score

        # Visual complexity
        complexity_score = 0.5
        if region_image is not None and region_image.size > 0:
            try:
                gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY)
                if gray.size > 0:
                    gradients = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
                    complexity = np.std(gradients) / 255.0
                    complexity_score = min(1.0, complexity * 2)
            except:
                pass
        score += 0.15 * complexity_score

        # Feature quality
        feature_quality = 0.5
        if features:
            total_quality = 0.0
            feature_count = 0

            if 'siglip' in features and features['siglip'] is not None:
                siglip_norm = torch.norm(features['siglip']).item()
                total_quality += min(1.0, siglip_norm) * 2.0
                feature_count += 2.0

            if 'dinov2' in features and features['dinov2'] is not None:
                dinov2_norm = torch.norm(features['dinov2']).item()
                total_quality += min(1.0, dinov2_norm)
                feature_count += 1

            if feature_count > 0:
                feature_quality = total_quality / feature_count

        score += 0.10 * feature_quality
        return min(1.0, score)

    def _passes_quality_check(self, landmark):
        """Quality check."""
        quality = landmark['quality_score']
        area = landmark['area']
        stability = landmark['stability_score']

        quality_threshold = self.quality_threshold * 0.9
        stability_threshold = 0.1

        return (quality >= quality_threshold and
                self.min_area_threshold <= area <= self.max_area_threshold and
                stability >= stability_threshold)

    def _filter_landmarks(self, landmarks):
        """Filter landmarks."""
        if not landmarks:
            return landmarks

        landmarks.sort(key=lambda x: x['quality_score'], reverse=True)

        filtered_landmarks = []
        for landmark in landmarks:
            keep = True
            for existing in filtered_landmarks:
                overlap = self._compute_spatial_overlap(landmark, existing)
                if overlap > 0.4:
                    keep = False
                    break
            if keep:
                filtered_landmarks.append(landmark)

        return filtered_landmarks[:self.max_landmarks_per_image]

    def _compute_spatial_overlap(self, landmark1, landmark2):
        """Compute spatial overlap."""
        bbox1 = landmark1['bbox']
        bbox2 = landmark2['bbox']

        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / max(union, 1)

    def get_session_directory(self):
        """Get current session landmark image directory."""
        return self.session_dir