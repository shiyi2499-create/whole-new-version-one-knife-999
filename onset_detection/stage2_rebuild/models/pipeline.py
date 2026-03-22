"""
End-to-end pipeline: runs Stage 2A → Stage 2B on a continuous IMU stream.

This module takes the output of Stage 1 (coarse password region) and produces
structured onset predictions ready for Stage 3 character classification.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from models.stage2a import GroupSegmentor
from models.stage2b import OnsetDetector
from utils.signal_processing import preprocess_imu
from utils.metrics import (
    compute_group_iou, compute_onset_metrics, compute_e2e_metrics,
    compute_full_report
)
from configs.config import PipelineConfig, Stage2AConfig, Stage2BConfig


class Stage2Pipeline:
    """
    End-to-end Stage 2 inference pipeline.

    Usage:
        pipeline = Stage2Pipeline.from_checkpoints(
            stage2a_ckpt='runs/stage2a/best.pt',
            stage2b_ckpt='runs/stage2b/best.pt',
            config=PipelineConfig(),
        )
        results = pipeline.run(coarse_region_imu, sample_rate=190)
    """

    def __init__(self,
                 stage2a_model: GroupSegmentor,
                 stage2b_model: OnsetDetector,
                 config: PipelineConfig,
                 device: torch.device = None):
        self.stage2a = stage2a_model
        self.stage2b = stage2b_model
        self.config = config
        self.device = device or torch.device('cpu')

        self.stage2a.to(self.device).eval()
        self.stage2b.to(self.device).eval()

    @classmethod
    def from_checkpoints(cls,
                         stage2a_ckpt: str,
                         stage2b_ckpt: str,
                         config: PipelineConfig,
                         device: str = 'auto'):
        """Load pipeline from saved checkpoints."""
        if device == 'auto':
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            dev = torch.device(device)

        # Load Stage 2A
        ckpt_2a = torch.load(stage2a_ckpt, map_location=dev, weights_only=False)
        stage2a_config = ckpt_2a.get('config', config.stage2a)
        if isinstance(stage2a_config, dict):
            stage2a_config = Stage2AConfig(**stage2a_config)
        stage2a_model = GroupSegmentor(stage2a_config)
        stage2a_model.load_state_dict(ckpt_2a['model_state_dict'])

        # Load Stage 2B
        ckpt_2b = torch.load(stage2b_ckpt, map_location=dev, weights_only=False)
        stage2b_config = ckpt_2b.get('config', config.stage2b)
        if isinstance(stage2b_config, dict):
            stage2b_config = Stage2BConfig(**stage2b_config)
        stage2b_model = OnsetDetector(stage2b_config)
        stage2b_model.load_state_dict(ckpt_2b['model_state_dict'])

        print(f"Loaded Stage 2A from {stage2a_ckpt}")
        print(f"Loaded Stage 2B from {stage2b_ckpt}")

        return cls(stage2a_model, stage2b_model, config, dev)

    def run(self,
            coarse_region_imu: np.ndarray,
            sample_rate: Optional[int] = None,
            ) -> Dict:
        """
        Run the full Stage 2 pipeline on a coarse password region.

        Args:
            coarse_region_imu: [T, 6] raw IMU data (accel_xyz + gyro_xyz)
                               This should be the region identified by Stage 1.
            sample_rate: override sample rate (default from config)

        Returns:
            {
                'group_boundaries': list of (start, end) in samples,
                'onset_positions': list of lists (per group),
                'group_probs': [T] stage2a probabilities,
                'onset_probs_per_group': list of [T_group] arrays,
                'num_groups': int,
                'onsets_per_group': int,
            }
        """
        sr = sample_rate or self.config.signal.sample_rate
        cfg_2a = self.config.stage2a
        cfg_2b = self.config.stage2b

        # Preprocess
        processed, stats = preprocess_imu(
            coarse_region_imu,
            sample_rate=sr,
            normalize=True,
            bandpass=True,
            bandpass_low=self.config.signal.bandpass_low,
            bandpass_high=self.config.signal.bandpass_high,
            add_magnitude=self.config.signal.use_magnitude,
        )

        # === Stage 2A: Group Segmentation ===
        x_2a = torch.from_numpy(processed.T).float().unsqueeze(0).to(self.device)  # [1, C, T]
        group_probs = self.stage2a.predict_probs(x_2a)[0].cpu().numpy()  # [T]

        group_boundaries = GroupSegmentor.post_process(
            group_probs,
            sample_rate=sr,
            median_kernel=cfg_2a.median_filter_kernel,
            threshold=cfg_2a.threshold,
            min_group_duration_s=cfg_2a.min_group_duration_s,
            expected_groups=cfg_2a.expected_groups,
        )

        # === Stage 2B: Onset Detection per group ===
        min_iki_samples = int(cfg_2b.min_iki_ms / 1000.0 * sr)
        all_onsets = []
        all_onset_probs = []

        for g_start, g_end in group_boundaries:
            group_signal = processed[g_start:g_end]

            if len(group_signal) < 5:
                # Too short, skip
                all_onsets.append(np.array([]))
                all_onset_probs.append(np.array([]))
                continue

            x_2b = torch.from_numpy(group_signal.T).float().unsqueeze(0).to(self.device)
            onset_probs = self.stage2b.predict_probs(x_2b)[0].cpu().numpy()

            # Pick peaks
            local_onsets = OnsetDetector.pick_peaks(
                onset_probs,
                expected_onsets=cfg_2b.expected_onsets,
                min_iki_samples=min_iki_samples,
                base_threshold=cfg_2b.peak_height_threshold,
                fallback_thresholds=cfg_2b.fallback_thresholds,
            )

            # Convert to global coordinates
            global_onsets = local_onsets + g_start
            all_onsets.append(global_onsets)
            all_onset_probs.append(onset_probs)

        return {
            'group_boundaries': group_boundaries,
            'onset_positions': all_onsets,
            'group_probs': group_probs,
            'onset_probs_per_group': all_onset_probs,
            'num_groups': len(group_boundaries),
            'onsets_per_group': cfg_2b.expected_onsets,
            'preprocess_stats': stats,
        }

    def evaluate_on_session(self,
                            coarse_region_imu: np.ndarray,
                            gt_group_boundaries: List[Tuple[int, int]],
                            gt_onset_positions: List[List[int]],
                            gt_chars: Optional[List[List[str]]] = None,
                            classifier_fn=None,
                            sample_rate: Optional[int] = None,
                            ) -> Dict:
        """
        Run pipeline and compute all metrics against ground truth.

        Args:
            coarse_region_imu: [T, 6] raw IMU
            gt_group_boundaries: list of (start, end) GT group boundaries
            gt_onset_positions: list of lists of GT onset positions per group
            gt_chars: optional GT characters per group
            classifier_fn: optional function(imu_window) → (pred_char, top_k_list)
            sample_rate: Hz

        Returns:
            Full metrics dict
        """
        sr = sample_rate or self.config.signal.sample_rate

        # Run pipeline
        results = self.run(coarse_region_imu, sample_rate=sr)

        # Stage 2A metrics
        stage2a_metrics = compute_group_iou(
            results['group_boundaries'], gt_group_boundaries
        )

        # Stage 2B metrics (per group)
        stage2b_metrics = []
        tolerance = int(0.05 * sr)  # 50ms

        n_groups = min(len(results['onset_positions']), len(gt_onset_positions))
        for g in range(n_groups):
            pred_ons = results['onset_positions'][g]
            gt_ons = np.array(gt_onset_positions[g])
            if len(gt_ons) > 0:
                m = compute_onset_metrics(pred_ons, gt_ons, tolerance, sr)
            else:
                m = {'precision': 0, 'recall': 0, 'f1': 0,
                     'mean_abs_error_ms': float('inf'), 'n_pred': len(pred_ons),
                     'n_gt': 0, 'n_matched': 0}
            stage2b_metrics.append(m)

        # E2E character metrics (if classifier available)
        e2e_metrics = {'char_top1': 0, 'CER': 1.0, 'exact_match_count': 0,
                       'total_groups': n_groups, 'total_chars': 0}

        if classifier_fn is not None and gt_chars is not None:
            pred_chars = []
            top_k_preds = []

            for g in range(n_groups):
                group_chars = []
                group_topk = []
                for onset_pos in results['onset_positions'][g]:
                    # Align with the current main-repo classifier protocol:
                    # 100 ms pre-trigger + 200 ms post-trigger.
                    pre = int(round(0.100 * sr))
                    post = int(round(0.200 * sr))
                    window_start = max(0, int(onset_pos) - pre)
                    window_end = min(len(coarse_region_imu), int(onset_pos) + post)
                    window = coarse_region_imu[window_start:window_end]

                    pred_char, topk = classifier_fn(window)
                    group_chars.append(pred_char)
                    group_topk.append(topk)

                pred_chars.append(group_chars)
                top_k_preds.append(group_topk)

            e2e_metrics = compute_e2e_metrics(pred_chars, gt_chars[:n_groups], top_k_preds)

        # Full report
        report = compute_full_report(stage2a_metrics, stage2b_metrics, e2e_metrics, sr)

        return {
            'stage2a_metrics': stage2a_metrics,
            'stage2b_metrics': stage2b_metrics,
            'e2e_metrics': e2e_metrics,
            'pipeline_results': results,
            'report': report,
        }
