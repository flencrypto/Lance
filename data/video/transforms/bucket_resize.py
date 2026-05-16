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

import math
from typing import List, Tuple, Union
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import InterpolationMode, to_tensor


class BucketResize:
    def __init__(
        self,
        max_area: float,
        interpolation: InterpolationMode = InterpolationMode.LANCZOS,
        aspect_ratios: List[str] = None,
        stride: Union[int, Tuple[int]] = None,
    ):
        self.max_area = max_area
        self.interpolation = interpolation

        assert aspect_ratios and stride, "`aspect_ratios` or `stride` not given!"
        self.buckets, self.bucket_ratios = self.init_buckets(aspect_ratios, max_area, stride)
        self.bucket_resize = {
            # NOTICE: 虽然名字叫 random, 但在这个 setting 下是 center crop, 无随机性
            # bucket: (h,w)
            bucket: RandomResizedCrop(
                size=(bucket[0], bucket[1]),
                scale=(1, 1),
                ratio=(bucket_ratio, bucket_ratio),
                interpolation=self.interpolation,
            )
            for bucket, bucket_ratio in zip(self.buckets, self.bucket_ratios)
        }

    def __call__(self, image: Union[torch.Tensor, Image.Image, List[Image.Image]]):

        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, list) and isinstance(image[0], Image.Image):
            width, height = image[0].size
        else:
            raise NotImplementedError

        bucket = self.find_nearest_bucket(width, height)
        resizer = self.bucket_resize[bucket]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            return torch.stack([to_tensor(resizer(_image)) for _image in image])
        else:
            image = resizer(image)
            if isinstance(image, Image.Image):
                image = to_tensor(image)
            return image

    def find_nearest_bucket(self, width, height):
        """
        找到与给定图片最近的bucket尺寸
        """
        image_ratio = width / height
        diff = np.abs(image_ratio - self.bucket_ratios)
        index = diff.argmin()
        return self.buckets[index]

    @staticmethod
    def init_buckets(aspect_ratio_names, max_area, stride):
        """
        指定一些列最接近给定宽高比和面积的,同时整除vae降采样和patch_size倍数的宽高
        """
        if not isinstance(stride, (tuple, list)):
            stride = (stride, stride)
        height_factor, width_factor = stride

        buckets, bucket_ratios = [], []
        for name in aspect_ratio_names:
            w, h = (int(v) for v in name.split(":"))
            aspect_ratio = w / h

            resize_width1 = math.sqrt(max_area * aspect_ratio)
            bucket_width1 = round(resize_width1 / width_factor) * width_factor
            resize_height1 = bucket_width1 / aspect_ratio
            bucket_height1 = round(resize_height1 / height_factor) * height_factor
            bucket_ratio1 = bucket_width1 / bucket_height1
            bucket_area1 = bucket_width1 * bucket_height1

            resize_height2 = math.sqrt(max_area / aspect_ratio)
            bucket_height2 = round(resize_height2 / height_factor) * height_factor
            resize_width2 = bucket_height2 * aspect_ratio
            bucket_width2 = round(resize_width2 / width_factor) * width_factor
            bucket_ratio2 = bucket_width2 / bucket_height2
            bucket_area2 = bucket_width2 * bucket_height2

            if abs(bucket_ratio1 - aspect_ratio) < abs(bucket_ratio2 - aspect_ratio):
                bucket_width, bucket_height = bucket_width1, bucket_height1
            elif abs(bucket_ratio1 - aspect_ratio) > abs(bucket_ratio2 - aspect_ratio):
                bucket_width, bucket_height = bucket_width2, bucket_height2
            else:
                if abs(bucket_area1 - max_area) <= abs(bucket_area2 - max_area):
                    bucket_width, bucket_height = bucket_width1, bucket_height1
                else:
                    bucket_width, bucket_height = bucket_width2, bucket_height2

            bucket_ratio = bucket_width / bucket_height

            buckets.append((bucket_height, bucket_width))
            bucket_ratios.append(bucket_ratio)

        bucket_ratios = np.array(bucket_ratios)

        return buckets, bucket_ratios


# ================================================================= #
# <<< 这里是为您编写的 check 函数 >>>
# ================================================================= #

def check_buckets(max_area: int, aspect_ratios: List[str], stride: int):
    """
    一个检查并打印 BucketResize.init_buckets 输出的辅助函数。

    Args:
        max_area (int): 目标总像素面积。
        aspect_ratios (List[str]): 目标宽高比列表 (例如: ["1:1", "4:3"])。
        stride (int): 步幅，高度和宽度必须是它的整数倍。
    """
    print(f"--- Checking Configuration ---")
    print(f"Max Area: {max_area} | Aspect Ratios: {aspect_ratios} | Stride: {stride}")
    print("-" * 35)

    buckets, bucket_ratios = BucketResize.init_buckets(aspect_ratios, max_area, stride)

    print("Generated Buckets (Height, Width) and Ratios:")
    for (h, w), ratio in zip(buckets, bucket_ratios):
        # 打印每个桶的尺寸、宽高比和总面积
        print(f"  - Bucket: ({h:4d}, {w:4d})  |  Ratio: {ratio:.4f}  |  Area: {h*w}")
    print("\n")


if __name__ == '__main__':
    # 示例1: 您提到的 256x256 的情况
    # 注意: max_area 是总像素，所以是 256*256
    check_buckets(
        # max_area=256*256,
        max_area=224*224,
        aspect_ratios=["21:9", '1:1', '4:3', '3:4', '9:16', '16:9'],
        stride=28 #16
    )

    # check_buckets(
    #     max_area=640*640,
    #     aspect_ratios=['1:1', '4:3', '3:4', '9:16', '16:9'],
    #     stride=16
    # )

    # check_buckets(
    #     max_area=512*512,
    #     aspect_ratios=['1:1', '4:3', '3:4', '9:16', '16:9'],
    #     stride=16
    # )

    # check_buckets(
    #     max_area=1024*1024,
    #     aspect_ratios=['1:1', '4:3', '3:4', '16:9', '9:16'],
    #     stride=16
    # )
