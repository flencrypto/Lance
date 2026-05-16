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

import os
import os.path as osp
import uuid
from typing import Any, Optional
import torch
from safetensors.torch import save_file as save_safetensors, save_model as save_safetensors_model
from .logging import get_logger
from .distributed import get_global_rank

logger = get_logger(__name__)

# 解决循环导入问题：延迟导入 is_hdfs_path, mkdir, copy
def _get_filesystem_funcs():
    from ..io.filesystem import is_hdfs_path, mkdir, copy
    return is_hdfs_path, mkdir, copy

_local_dir = None


def get_local_dir():
    """
    Get a local directory for temporary storage for this process.
    """
    global _local_dir
    _, mkdir, _ = _get_filesystem_funcs()
    if _local_dir is None:
        _local_dir = os.path.join("persistence", "rank_" + str(get_global_rank()) + "_" + str(uuid.uuid4()))
        mkdir(_local_dir)
    return _local_dir


def set_local_dir(dirname):
    """
    Set a local directory for temporary storage for this process.
    """
    global _local_dir
    _, mkdir, _ = _get_filesystem_funcs()
    if dirname is None:
        return
    _local_dir = os.path.join(dirname, str(uuid.uuid4()))
    mkdir(_local_dir)


def get_local_path(path: str) -> str:
    """
    Get a local path for storing the file.
    If the path is already a local path, directly return.
    """
    is_hdfs_path, mkdir, _ = _get_filesystem_funcs()
    if is_hdfs_path(path):
        path = os.path.join(get_local_dir(), os.path.basename(path))
    else:
        mkdir(os.path.dirname(path))
    return path


def convert_dtype(states: Any, dtype: Optional[torch.dtype] = None):
    """
    Recursively convert the state_dict to device and dtype.
    """
    if dtype is None:
        return states
    if torch.is_tensor(states):
        return states.to("cpu", dtype)
    if isinstance(states, dict):
        return {k: convert_dtype(v, dtype) for k, v in states.items()}
    if isinstance(states, list):
        return [convert_dtype(v, dtype) for v in states]
    return states


def save(data: Any, path: str, blocking: bool = True, persistence_dir: Optional[str] = None):
    """
    安全地将数据保存到指定路径（本地或HDFS）。
    此版本使用 get_local_dir 来处理临时文件。
    """
    is_hdfs_path, _, copy = _get_filesystem_funcs()
    if not is_hdfs_path(path):
        if path.endswith(".safetensors"):
            if isinstance(data, torch.nn.Module):
                save_safetensors_model(data, path)
            else:
                save_safetensors(data, path)
        else:
            torch.save(data, path)

        logger.info(f"Early saved to local path: {path}")
        return

    # --- HDFS 路径处理 ---
    # 1. 获取一个唯一的本地临时文件路径
    if persistence_dir is None:
        persistence_dir = get_local_dir()

    try:
        # 2. 向临时文件写入数据
        local_path = osp.join(persistence_dir, osp.basename(path))
        if path.endswith(".safetensors"):
            if isinstance(data, torch.nn.Module):
                save_safetensors_model(data, local_path)
            else:
                save_safetensors(data, local_path)
        else:
            torch.save(data, local_path)
        logger.info(f"Saved to local path: {local_path}")

        # 3. 将本地临时文件复制到HDFS
        copy(local_path, path, blocking=blocking)
        logger.info(f"Copy {local_path} to HDFS or Local path: {path} done.")

    finally:
        # NOTE: 因为是重复写入，不需要清理了
        pass

        # # 4. 清理临时文件
        # # NOTE: 暂时只在blocking为True的时候清理
        # if osp.exists(persistence_path) and blocking:
        #     os.remove(persistence_path)
        #     logger.info(f"Removed temporary file: {persistence_path}")

def dummy_indexes_searchsorted(packed_text_indexes: torch.LongTensor, ce_loss_indexes: torch.LongTensor) -> torch.LongTensor:
    """
    使用 searchsorted 方法：
    - 对 packed_text_indexes 排序，得到排序值 sorted_vals 和原始下标 sorted_pos。
    - 在 sorted_vals 中查找 ce_loss_indexes 的位置 loc。
    - 根据 loc 索引 sorted_pos，得到 dummy_indexes。
    """
    sorted_vals, sorted_pos = torch.sort(packed_text_indexes)
    loc = torch.searchsorted(sorted_vals, ce_loss_indexes)
    return sorted_pos[loc]
