# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding: utf-8

"""
Frame samplers.

TODO: 可能需要写一下满足自定义需求的frame sampler
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, Union
import numpy as np


class FrameSamplerOutput(NamedTuple):
    """
    Return indices for frame decoding,
    and optionally additional information to return to user.
    """

    indices: List[int]
    additional_info: Dict[str, Any] = {}


class FrameSampler(ABC):
    """
    Frame sampler base class.

    Child class must implement __call__ method to return the decoding indices.
    Or raise if the video cannot be sampled (e.g. too short, etc.)
    """

    @abstractmethod
    def __call__(self, num_frames: int) -> FrameSamplerOutput:
        raise NotImplementedError


class AllFrameSampler(FrameSampler):
    """
    All frame sampler. Returns all frames in a video.
    """

    def __call__(self, num_frames: int) -> FrameSamplerOutput:
        return FrameSamplerOutput(indices=list(range(num_frames)))

class OnlyFirstFrameSampler:
    """
    Only first frame sampler. Returns only the first frame of a video.
    """

    def __call__(self, frames_info: Dict[str, int], **kwargs) -> FrameSamplerOutput:
        return FrameSamplerOutput(indices=[0])

class FixedFrameSampler:
    """
    固定帧数采样器（上/下采样统一算法）：
    - 接受包含 start_frame, end_frame, total_frames 的 frames_info dict；
    - 对任意 total_frames ≥ 1，总是返回长度为 num_frames 的帧编号列表；
    - 保证首尾对应 start_frame 和 end_frame - 1，内部等距离分布；
    - 当 total_frames < num_frames 时会重复索引，如 [0,1,2] → [0,0,1,1,2,2]。
    """
    def __init__(self, num_frames: int):
        if num_frames < 1:
            raise ValueError("num_frames must be ≥ 1")
        self.num_frames = num_frames

    def __call__(self, frames_info: Dict[str, int]) -> List[int]:
        """
        参数:
            frames_info: 包含 'start_frame', 'end_frame', 'total_frames' 的字典
        返回:
            List[int]: 采样后的全局帧编号列表，长度恒为 num_frames
        """
        start = frames_info.get('start_frame')
        total = frames_info.get('total_frames')
        end = frames_info.get('end_frame')
        if start is None or total is None or end is None:
            raise ValueError("frames_info must contain 'start_frame', 'end_frame', and 'total_frames'")
        if total < 1:
            raise ValueError("total_frames must be ≥ 1")
        # 计算相对索引
        rel_indices = self._get_indices(total)
        # 转换为全局并确保不越界
        indices = [min(start + idx, end - 1) for idx in rel_indices]

        return FrameSamplerOutput(
            indices=indices,
            additional_info={
                "start_frame": start,
                "end_frame": end,
                "total_frames": total,
            },
        )

    def _get_indices(self, total: int) -> List[int]:
        # 单帧特殊处理
        if self.num_frames == 1:
            return [0]
        # 统一采样公式，包括上采样和下采样场景
        return [
            int(round(i * (total - 1) / (self.num_frames - 1)))
            for i in range(self.num_frames)
        ]


class ConsecutiveFrameSampler(FrameSampler):
    """
    Adaptive frame sampler.

    Arguments:
        stride: frame skip.
                For example, 1 denotes no skip. 2 denotes select every other frame. 3
                denotes select every third frame. When a list is given, stride is randomly
                chosen with even probability. However, user may set it to [1,1,2] to
                denote 1 with 66% probability and 2 with 33% proability.
        clip:   clip location.
                    "center":   clip video at the center.
                    "uniform":  clip video uniformly at random.
        jitter: jitter to the location.
                Only applicable when clip is "center".
                The value is the stdev of the normal distribution to shift the index.
    """

    def __init__(
        self,
        strides: Union[int, List[int]] = 1,
        temporal: int = 4,
        clip: Literal["center", "uniform"] = "uniform",
        jitter: float = 0.0,
    ):
        strides = [strides] if isinstance(strides, int) else strides
        assert len(strides) > 0
        self.strides = np.array(strides)
        self.temporal = temporal
        self.clip = clip
        self.jitter = jitter

    def __call__(self, frames_info: Dict[str, int]) -> FrameSamplerOutput:

        start_frame = frames_info["start_frame"]
        end_frame = frames_info["end_frame"]
        num_frames = frames_info["total_frames"]

        stride = np.random.choice(self.strides)

        frames = end_frame - start_frame
        length = frames // stride

        # Calculate the maximum integer of the form kn + 1 that does not exceed the given length.
        def _max_kn_plus_1(length, k):
            if length < 1:
                raise ValueError("Length must be at least 1.")
            n = (length - 1) // k
            return k * n + 1

        length = _max_kn_plus_1(length, self.temporal)

        # Choose start index.
        min_start_index = start_frame
        max_start_index = end_frame - 1 - stride * (length - 1)

        mid_start_index = round((min_start_index + max_start_index) / 2)
        jitter = round(np.random.normal(loc=0, scale=self.jitter))

        if self.clip == "head":
            start_index = min_start_index
        elif self.clip == "tail":
            start_index = max_start_index
        elif self.clip == "center":
            start_index = mid_start_index + jitter
        elif self.clip == "uniform":
            start_index = np.random.randint(min_start_index, max_start_index + 1)
        else:
            raise NotImplementedError

        start_index = np.clip(start_index, min_start_index, max_start_index)

        # Compute indices
        indices = np.arange(start_index, start_index + length * stride, stride)

        # Return indices and additional information to return to user.
        return FrameSamplerOutput(
            indices=indices.tolist(),
            additional_info={
                "stride": stride,
                "start_frame": start_index,
                "end_frame": start_index + length * stride,
                "total_frames": num_frames,
            },
        )


class AdaptiveFrameSampler(FrameSampler):
    """
    Adaptive frame sampler.

    Arguments:
        length: frame length to return.
                For example, [5,10] denotes to always return 5 frames or 10 frames.
                It will choose the longest length that fits the original video.
                For example, if the video is 9 frames total, it will clip to 5 frames.
        stride: frame skip.
                For example, 1 denotes no skip. 2 denotes select every other frame. 3
                denotes select every third frame. When a list is given, stride is randomly
                chosen with even probability. However, user may set it to [1,1,2] to
                denote 1 with 66% probability and 2 with 33% proability.
        clip:   clip location.
                    "center":   clip video at the center.
                    "uniform":  clip video uniformly at random.
        jitter: jitter to the location.
                Only applicable when clip is "center".
                The value is the stdev of the normal distribution to shift the index.
    """

    def __init__(
        self,
        lengths: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        clip: Literal["center", "uniform"] = "uniform",
        jitter: float = 0.0,
    ):
        lengths = [lengths] if isinstance(lengths, int) else lengths
        strides = [strides] if isinstance(strides, int) else strides
        assert len(lengths) > 0
        assert len(strides) > 0
        assert clip in ["center", "uniform"]
        assert jitter >= 0
        self.lengths = np.array(lengths)
        self.strides = np.array(strides)
        self.clip = clip
        self.jitter = jitter

    def __call__(
        self,
        num_frames: int,
    ) -> FrameSamplerOutput:
        # Choose stride.
        # Drop strides that are too long for this video.
        # Then randomly choose a valid stride.
        valid_strides = np.any(num_frames // self.strides >= self.lengths.reshape(-1, 1), axis=0)
        valid_strides = self.strides[valid_strides]
        if valid_strides.size <= 0:
            raise ValueError(f"Video is too short ({num_frames} frames).")
        stride = np.random.choice(valid_strides)

        # Choose length.
        # Pick the max length that can fit the video under the current stride.
        valid_lengths = self.lengths[num_frames // stride >= self.lengths]
        length = np.max(valid_lengths)

        # Choose start index.
        min_start_index = 0
        max_start_index = num_frames - 1 - stride * (length - 1)
        mid_start_index = round((min_start_index + max_start_index) / 2)
        jitter = round(np.random.normal(loc=0, scale=self.jitter))

        if self.clip == "center":
            start_index = mid_start_index + jitter
        elif self.clip == "uniform":
            start_index = np.random.randint(min_start_index, max_start_index + 1)
        else:
            raise NotImplementedError

        start_index = np.clip(start_index, min_start_index, max_start_index)

        # Compute indices
        indices = np.arange(start_index, start_index + length * stride, stride)

        # Return indices and additional information to return to user.
        return FrameSamplerOutput(
            indices=indices.tolist(),
            additional_info={
                "stride": stride,
                "start_frame": start_index,
                "end_frame": start_index + length * stride,
                "total_frames": num_frames,
            },
        )


@dataclass
class AdaptiveAdvancedFrameSamplerStrategy:
    stride: int
    stride_prob: float
    frame_lengths: List[int]
    frame_lengths_prob: Union[Literal["uniform", "harmonic"], List[float]]


class AdaptiveAdvancedFrameSampler(FrameSampler):
    """
    Advanced adaptive frame sampler supports different frame lengths for different strides,
    and supports probabilistic sampling of both the stride and the frame length.

    strategies: A list of strategies to sample from.
    clip:   clip location.
            "center":   clip video at the center.
            "uniform":  clip video uniformly at random.
    jitter: jitter to the location.
            Only applicable when clip is "center".
            The value is the stdev of the normal distribution to shift the index.
    """

    def __init__(
        self,
        strategies: List[AdaptiveAdvancedFrameSamplerStrategy],
        clip: Literal["center", "uniform"] = "uniform",
        jitter: float = 0.0,
    ):
        assert len(strategies) > 0, "Strategies must not be empty"
        assert len({s.stride for s in strategies}) == len(strategies), "Strides cannot duplicate."
        assert clip in ["center", "uniform"]
        assert jitter >= 0
        self.clip = clip
        self.jitter = jitter
        self.strides = []
        self.strides_prob = []
        self.frame_lengths = []
        self.frame_lengths_prob = []

        for strategy in sorted(strategies, key=lambda s: s.stride):
            # Validate strides.
            assert isinstance(strategy.stride, int), "Stride must be an integer."
            assert strategy.stride > 0, "Stride must be a positive integer."
            self.strides.append(strategy.stride)

            # Assign strides_prob.
            assert isinstance(strategy.stride_prob, (int, float)), "Stride prob is not int/float."
            assert strategy.stride_prob >= 0, "Stride prob must be non-negative."
            self.strides_prob.append(strategy.stride_prob)

            # Assign frame lengths, sort by value.
            assert len(strategy.frame_lengths) > 0, "Frame lengths must not be empty."
            frame_lengths = np.array(strategy.frame_lengths)
            assert frame_lengths.dtype == int, "Frame lengths must be integers."
            assert np.all(frame_lengths > 0), "Frame lengths must be positive integers."
            frame_lengths_sorted_idx = np.argsort(frame_lengths)
            frame_lengths = frame_lengths[frame_lengths_sorted_idx]
            self.frame_lengths.append(frame_lengths)

            # Assign frame lengths prob, apply the sorting to prob as well.
            if strategy.frame_lengths_prob == "uniform":
                # e.g. [0.2, 0.2, 0.2, 0.2, 0.2]
                frame_lengths_prob = np.full(len(frame_lengths), 1.0 / len(frame_lengths))
            elif strategy.frame_lengths_prob == "harmonic":
                # e.g. [0.2, 0.25, 0.33, 0.5, 1]
                frame_lengths_prob = np.flip(1 / np.arange(1, len(frame_lengths) + 1))
            elif isinstance(strategy.frame_lengths_prob, list):
                frame_lengths_prob = np.array(strategy.frame_lengths_prob)
                frame_lengths_prob = frame_lengths_prob[frame_lengths_sorted_idx]
            else:
                raise NotImplementedError
            assert len(frame_lengths_prob) == len(frame_lengths), "Frame lengths prob mismatch."
            assert np.all(frame_lengths_prob >= 0), "Frame lengths prob must not be negative."
            assert frame_lengths_prob.sum() > 0, "Frame lengths prob must not be all zeros."
            frame_lengths_prob /= frame_lengths_prob.sum()
            self.frame_lengths_prob.append(frame_lengths_prob)

        self.strides = np.array(self.strides)
        self.strides_prob = np.array(self.strides_prob)
        assert self.strides_prob.sum() > 0, "Strides prob must not be all zeros."
        self.strides_prob /= self.strides_prob.sum()

    def __call__(self, num_frames: int):
        sample_result = adptive_sample_framelen_and_stride(
            num_frames=num_frames,
            strides=self.strides,
            strides_prob=self.strides_prob,
            frame_lengths=self.frame_lengths,
            frame_lengths_prob=self.frame_lengths_prob,
        )

        stride = sample_result["stride"]
        length = sample_result["frame_length"]

        # Choose start index.
        min_start_index = 0
        max_start_index = num_frames - 1 - stride * (length - 1)
        mid_start_index = round((min_start_index + max_start_index) / 2)
        jitter = round(np.random.normal(loc=0, scale=self.jitter))

        if self.clip == "center":
            start_index = mid_start_index + jitter
        elif self.clip == "uniform":
            start_index = np.random.randint(min_start_index, max_start_index + 1)
        else:
            raise NotImplementedError

        start_index = np.clip(start_index, min_start_index, max_start_index)

        # Compute indices
        indices = np.arange(start_index, start_index + length * stride, stride)

        # Return indices and additional information to return to user.
        return FrameSamplerOutput(
            indices=indices.tolist(),
            additional_info={
                "stride": stride,
                "start_frame": start_index,
                "end_frame": start_index + length * stride,
                "total_frames": num_frames,
            },
        )


class MultiClipsFrameSampler(FrameSampler):
    """
    multi clips frame sampler.

    Arguments:
        temporal: downsample factor on temporal
        sample_fps: fps of sampled frames
        truncate: whether to truncate by max duration of the video (default = false, already truncate in clip_indices)
        max_duration: truncate by max duration of the video
    """

    def __init__(
        self,
        temporal: int = 4,
        sample_fps: int = 12,
        truncate: bool = False,
        max_duration: int = 12,
        length_type: Literal["kn", "kn+1"] = "kn+1",
        assert_seconds: bool = True,
    ):
        self.temporal = temporal
        self.sample_fps = sample_fps
        self.truncate = truncate
        self.max_duration = max_duration
        self.length_type = length_type
        self.assert_seconds = assert_seconds

    def __call__(self, frames_info: Dict[str, int]) -> FrameSamplerOutput:

        clip_indices = frames_info["clip_indices"]
        origin_fps = frames_info["fps"]

        if self.truncate:
            clip_indices = self.truncate_to_bucket(clip_indices, origin_fps)

        if self.assert_seconds:
            duration_sec = int(round(sum([(end - start) / origin_fps for start, end in clip_indices])))
            if not self.truncate:                         # 新增：即使不截段也限制总时长
                duration_sec = min(duration_sec, self.max_duration)
            duration = int(round(duration_sec))

            n_frames = duration * self.sample_fps
            if self.length_type == "kn+1":
                n_frames += 1
        else:
            duration = sum([(end - start) / origin_fps for start, end in clip_indices])
            if not self.truncate:                         # 新增
                duration = min(duration, self.max_duration)
            n_frames = int(round(duration * self.sample_fps))
            if self.length_type == "kn+1":
                if n_frames % self.temporal != 0:
                    n_frames = n_frames // self.temporal * self.temporal + 1
                else:
                    n_frames = n_frames // self.temporal * self.temporal + 1 - self.temporal
        clip_n_frames = self.split_n_frames_by_clip(n_frames, clip_indices)
        sample_indices = self.sample_frame_indices(clip_indices, clip_n_frames)

        clip_n_latent_frames = [(n + self.temporal - 1) // self.temporal for n in clip_n_frames]

        return FrameSamplerOutput(
            indices=sample_indices,
            additional_info={
                "clip_n_frames": clip_n_frames,
                "clip_n_latent_frames": clip_n_latent_frames,
            },
        )

    def truncate_to_bucket(self, clip_indices, fps):
        clip_indices = [tuple(index) for index in clip_indices]
        durations = []
        for start, end in clip_indices:
            durations.append((end - start) / fps)
        duration = sum(durations)
        max_duration = min(int(duration), self.max_duration)
        cutoff = duration - max_duration
        if cutoff > 0:
            if durations[-1] - cutoff > durations[0] - cutoff:  # 截掉尾部
                start, end = clip_indices[-1]
                end = min(round((durations[-1] - cutoff) * fps), end) + start
                clip_indices[-1] = (start, end)
            else:
                start, end = clip_indices[0]
                start = max(end - round((durations[0] - cutoff) * fps), start)
                clip_indices[0] = (start, end)
        return clip_indices

    def split_n_frames_by_clip(self, n_frames, clip_indices):
        n_latent_frames = n_frames // self.temporal
        clip_lengths = [(end - start) for start, end in clip_indices]
        clip_n_latent_frames = [int(l / sum(clip_lengths) * n_latent_frames) for l in clip_lengths]
        n_remains = n_latent_frames - sum(clip_n_latent_frames)
        for i in range(n_remains):
            clip_n_latent_frames[i] += 1
        clip_n_frames = [n * self.temporal for n in clip_n_latent_frames]
        if self.length_type == "kn+1":
            clip_n_frames[0] += 1
        return clip_n_frames

    def sample_frame_indices(self, clip_indices, clip_n_frames):
        shift_clip_indices = []
        accum_n_frames = 0
        for start, end in clip_indices:
            start, end = accum_n_frames, accum_n_frames + (end - start)
            shift_clip_indices.append((start, end))
            accum_n_frames += end - start

        all_sample_indices = []
        for i, ((start, end), (shift_start, shift_end), n_frames) in enumerate(
            zip(clip_indices, shift_clip_indices, clip_n_frames)
        ):
            indices = np.arange(start, end)
            next_shift_start = (
                shift_clip_indices[i + 1][0] if i < len(clip_indices) - 1 else shift_end
            )
            shift_sample_indices = (
                np.linspace(shift_start, next_shift_start - 1, n_frames, dtype=int) - shift_start
            )
            sample_indices = indices[shift_sample_indices].tolist()
            all_sample_indices.extend(sample_indices)

        return all_sample_indices


def normalize_probabilities(
    items: np.ndarray,
    probs: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(items), "Items must not be empty."
    assert len(items) == len(masks) == len(probs), "Lengths must match."
    # assert (items, np.ndarray), "isinstanceItems must be an np.ndarray."
    assert isinstance(probs, np.ndarray), "Probs must be an np.ndarray."
    assert isinstance(masks, np.ndarray), "Masks must be an np.ndarray."
    assert masks.dtype == bool, "Masks must be boolean."
    assert np.any(masks), "Masks must not be all False."
    assert np.all(np.diff(masks.astype("int")) <= 0), "Masks must not break monotonicity."

    ret_items = items[masks]
    ret_probs = probs[masks]

    # Accumulate the probabilities of infeasible items to the last feasible one.
    ret_probs[-1] += probs[~masks].sum()

    return ret_items, ret_probs


def adptive_sample_framelen_and_stride(
    num_frames: int,
    strides: np.ndarray,
    strides_prob: np.ndarray,
    frame_lengths: List[np.ndarray],
    frame_lengths_prob: List[Optional[np.ndarray]],
) -> Dict[str, Any]:
    """Adaptively sample frame length and stride for a video.

    Args:
        num_frames: Number of frames in the current video.
        strides: A list of strides.
        strides_prob: The probability for each stride.
        frame_lengths: The number of frames (sorted) to sample from at the current stride.
            For example, `frame_length=10` at `stride=2` means that we need to have 20 frames.
            When the number of frames to sample is infeasible, it will select the feasible frame
            lengths and re-normalize the probability according to the feasible frames at hand.
            For example, if `num_frames=10`, `frame_lengths[stride2]=[4, 5]`,
            `frame_lengths[stride3]=[1, 3, 5]`, we can sample frame lengths 1, 2, and 5 at
            `stride=2` (2, 4, and 10 frames) but only frame lengths 1, 3 at `stride=3`. In this
            case, we will add the probability of `frame_length=5` at `stride=3` to `frame_length=3`
            at `stride=3`, making it more likely to be selected.
        frame_lengths_prob: The frame probabilities to sample from the corresponding frame lengths.
            Defaults to None for uniform sampling.
    Returns:
        dictionary: A dictionary containing the selected frames and strides. if none is feasible,
        it will raise an exception.
    """
    assert len(strides) == len(strides_prob) == len(frame_lengths) == len(frame_lengths_prob)

    # Prepare frame_lengths_mask for each stride.
    frame_lengths_mask = [num_frames // s >= l for s, l in zip(strides, frame_lengths)]

    # Prepare stride mask and prob.
    strides_idxs = np.arange(len(strides))
    strides_mask = np.array([np.any(mask) for mask in frame_lengths_mask])
    assert np.any(strides_mask), (
        f"Cannot sample frames={num_frames} "
        + f"from strides={strides} and lengths={frame_lengths}"
    )

    # Drop infeasible strides and normalize probability.
    strides_idxs, strides_prob = normalize_probabilities(strides_idxs, strides_prob, strides_mask)

    # Choose stride.
    stride_idx = np.random.choice(strides_idxs, p=strides_prob)
    stride = strides[stride_idx]

    # Prepare frame_lengths mask and prob for the current stride.
    lengths = frame_lengths[stride_idx]
    lengths_mask = frame_lengths_mask[stride_idx]
    lengths_prob = frame_lengths_prob[stride_idx]
    if lengths_prob is None:
        lengths_prob = np.full(len(lengths), 1.0 / len(lengths))

    # Drop infeasible lengths and normalize probability.
    lengths, lengths_prob = normalize_probabilities(lengths, lengths_prob, lengths_mask)

    # Choose frame length.
    length = np.random.choice(lengths, p=lengths_prob)
    return dict(stride=stride, frame_length=length)
