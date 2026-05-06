#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_sam3d_skill_interfaces.py

测试 sam3d_grasp_core.py 里的两个接口：

1. files:
   测试 grasp_from_files(...)
   用已有 RGB / Depth / Audio 或 Text 跑。

2. current_mock:
   测试 grasp_current_scene(...)
   但暂时用 MockSceneProvider / MockVoiceProvider 模拟当前相机和麦克风。
"""

import argparse
import json

from sam3d_grasp_core import (
    grasp_from_files,
    grasp_current_scene,
    MockSceneProvider,
    MockVoiceProvider,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["files", "current_mock"],
        required=True,
        help="files=离线文件接口；current_mock=模拟当前场景接口",
    )

    parser.add_argument("--rgb", required=True, help="RGB 图片路径")
    parser.add_argument("--depth", required=True, help="Depth 深度图路径")
    parser.add_argument("--audio", default=None, help="语音文件路径")
    parser.add_argument("--text", default=None, help="文字指令")
    parser.add_argument("--out-root", default=None, help="输出目录")

    args = parser.parse_args()

    if args.mode == "files":
        result = grasp_from_files(
            rgb_path=args.rgb,
            depth_path=args.depth,
            audio_path=args.audio,
            text_command=args.text,
            out_root=args.out_root,
        )

    else:
        scene_provider = MockSceneProvider(
            rgb_path=args.rgb,
            depth_path=args.depth,
        )

        voice_provider = None

        if args.audio:
            voice_provider = MockVoiceProvider(
                audio_path=args.audio,
            )

        result = grasp_current_scene(
            scene_provider=scene_provider,
            voice_provider=voice_provider,
            text_command=args.text,
            out_root=args.out_root,
        )

    print("\n" + "=" * 100)
    print("Skill Result")
    print("=" * 100)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
