#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
modules/sam3d_engine_v2.py

这个文件专门给 pipeline4.py 使用。

目的：
  保留原始 modules/sam3d_engine.py 不动，
  在 v2 里包装原来的 SAM3DEngine，让 generate_3d 支持 keep_loaded 参数。

为什么需要：
  pipeline4 需要同一个 SAM3DEngine 连续生成两次：
    1. single object
    2. joint object + table

  原始 SAM3DEngine.generate_3d() 通常在生成结束后会调用 self._unload_model()，
  导致下一次生成又重新加载 10GB 级 SAM3D 权重。

  这个 v2 包装类通过重写 _unload_model()，在 keep_loaded=True 时跳过卸载。
  这样可以：
    第一次 generate_3d(..., keep_loaded=True) 生成 single 后保留权重；
    第二次 generate_3d(..., keep_loaded=False) 生成 joint 后正常释放权重。

使用方式：
  在 pipeline4.py 里把：
    from modules.sam3d_engine import SAM3DEngine
  改成：
    from modules.sam3d_engine_v2 import SAM3DEngine
"""

import gc
import torch

from modules.sam3d_engine import SAM3DEngine as _BaseSAM3DEngine


class SAM3DEngine(_BaseSAM3DEngine):
    """
    对原始 SAM3DEngine 的轻量包装。

    不复制原始 generate_3d 的复杂逻辑；
    只增加 keep_loaded 参数，并通过 _unload_model 拦截卸载行为。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 当前这次 generate_3d 是否需要保留模型
        self._pipeline4_keep_loaded = False

    def generate_3d(self, image_path, mask, keep_loaded=False, *args, **kwargs):
        """
        兼容原始接口：
          generate_3d(image_path, mask)

        新增接口：
          generate_3d(image_path, mask, keep_loaded=True)

        参数：
          keep_loaded=False:
            保持原始行为，生成后正常卸载 SAM3D 模型。

          keep_loaded=True:
            本次生成结束时，如果原始代码调用 self._unload_model()，
            v2 会拦截并跳过卸载，保留权重给下一次生成使用。
        """
        self._pipeline4_keep_loaded = bool(keep_loaded)

        print(
            f"[SAM-3D-V2] generate_3d called, "
            f"keep_loaded={self._pipeline4_keep_loaded}"
        )

        try:
            # 调用原始 SAM3DEngine.generate_3d
            # 注意：不要把 keep_loaded 继续传给 super，
            # 因为原始 generate_3d 不认识这个参数。
            return super().generate_3d(image_path, mask, *args, **kwargs)

        finally:
            # 这里不要主动卸载。
            # 原始 generate_3d 内部如果调用 self._unload_model()，
            # 会走到下面我们重写的 _unload_model()。
            #
            # 生成结束后把标记恢复，避免影响后续显式卸载。
            self._pipeline4_keep_loaded = False

    def _unload_model(self):
        """
        拦截原始 SAM3DEngine 里的卸载逻辑。

        当 generate_3d(..., keep_loaded=True) 时：
          跳过卸载，模型继续保留在显存中。

        当 generate_3d(..., keep_loaded=False) 时：
          调用原始 _unload_model()，正常释放显存。
        """
        if getattr(self, "_pipeline4_keep_loaded", False):
            print(
                "[SAM-3D-V2] keep_loaded=True，"
                "本次跳过 _unload_model()，继续复用 SAM3D 权重。"
            )
            return

        print("[SAM-3D-V2] keep_loaded=False，正常卸载 SAM3D 模型。")
        return super()._unload_model()

    def force_unload(self):
        """
        强制释放模型。

        用于异常退出、手动清理，忽略 keep_loaded 标记。
        """
        old_flag = getattr(self, "_pipeline4_keep_loaded", False)

        try:
            self._pipeline4_keep_loaded = False
            return super()._unload_model()
        finally:
            self._pipeline4_keep_loaded = old_flag

    def close(self):
        self.force_unload()

    def __del__(self):
        try:
            self.force_unload()
        except Exception:
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
