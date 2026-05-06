#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_dimos_sam3d_grasp_skill.py

启动 SAM3DContactGraspSkill，让 DimOS / MCP / Agent 可以调用。

兼容说明：
  当前 dimos==0.0.11 的 Blueprint.build() 不接受 global_config 参数，
  所以这里不再传 global_config。
"""

import time
import inspect

from dimos_sam3d_grasp_skill import SAM3DContactGraspSkill


def main():
    print("=" * 80)
    print("启动 SAM3DContactGraspSkill DimOS Skill")
    print("=" * 80)

    blueprint = SAM3DContactGraspSkill.blueprint()

    print("[DimOS] blueprint:", blueprint)

    # 兼容不同 DimOS 版本的 build() 签名
    build_sig = inspect.signature(blueprint.build)
    print("[DimOS] Blueprint.build signature:", build_sig)

    if "global_config" in build_sig.parameters:
        try:
            from dimos.core.global_config import GlobalConfig
            coordinator = blueprint.build(global_config=GlobalConfig())
        except Exception as e:
            print(f"[DimOS] 使用 global_config build 失败，改用无参数 build: {e}")
            coordinator = blueprint.build()
    else:
        coordinator = blueprint.build()

    print("[DimOS] coordinator:", coordinator)
    print("=" * 80)
    print("SAM3DContactGraspSkill 已启动")
    print("=" * 80)

    # 兼容不同版本的启动方法
    if hasattr(coordinator, "loop"):
        coordinator.loop()
    elif hasattr(coordinator, "run"):
        coordinator.run()
    elif hasattr(coordinator, "serve"):
        coordinator.serve()
    else:
        print("[DimOS] coordinator 没有 loop/run/serve 方法。")
        print("[DimOS] 当前对象类型:", type(coordinator))
        print("[DimOS] 当前对象属性:", dir(coordinator))
        print("[DimOS] 进入保活状态，按 Ctrl+C 退出。")
        while True:
            time.sleep(10)


if __name__ == "__main__":
    main()