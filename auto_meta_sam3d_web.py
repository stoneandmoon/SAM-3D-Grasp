#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_meta_sam3d_web.py

目标：
  在纯 SSH / headless Chromium 环境下，自动控制 Meta SAM3D 官网 demo：

    1. 打开官网
    2. 上传 RGB 图像
    3. 根据本地 mask 自动采样正点/负点
    4. 在网页图像上模拟点击
    5. 点击 Generate / 生成 3D
    6. 等待并点击 Download / 下载
    7. 自动保存下载文件
    8. 可选转换成 PLY

注意：
  这是浏览器自动化，不是官方稳定 API。
  如果官网 UI 改了，按钮文字或 DOM 选择器可能需要修改。

典型运行：

python auto_meta_sam3d_web.py \
  --rgb ./data/test/000002/rgb/000000.png \
  --mask ./output_duck_table_joint_mask/duck_plus_local_table_mask.png \
  --out-dir ./output_meta_web_auto \
  --download-dir ./meta_web_downloads \
  --num-pos 16 \
  --num-neg 8 \
  --timeout-sec 1800
"""

import os
import time
import json
import argparse
import shutil
import zipfile
import asyncio

import cv2
import numpy as np


META_SAM3D_URL = "https://aidemos.meta.com/segment-anything/editor/convert-image-to-3d"


# ============================================================
# 基础工具
# ============================================================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def load_image_size(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取 image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def load_mask(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"无法读取 mask: {mask_path}")
    return (m > 127).astype(np.uint8)


async def save_screenshot(page, out_dir, name):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, name)
    try:
        await page.screenshot(path=path, full_page=True)
        print(f"[Screenshot] {path}")
    except Exception as e:
        print(f"[Screenshot][WARN] 保存截图失败 {path}: {repr(e)}")


# ============================================================
# mask -> 点击点
# ============================================================

def sample_prompt_points(mask, num_pos=8, num_neg=8, min_dist=18):
    """
    从 mask 内部采样 positive points，
    从 mask 外但靠近 bbox 的区域采样 negative points。

    返回：
      positives: [(x, y), ...]
      negatives: [(x, y), ...]
    """
    H, W = mask.shape[:2]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("mask 为空，无法采样点击点")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    mask_u8 = (mask > 0).astype(np.uint8)

    # 正点：distance transform 取 mask 内部更中心的点
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    pos_candidates = np.column_stack(np.where(mask_u8 > 0))  # y, x
    pos_scores = dist_in[pos_candidates[:, 0], pos_candidates[:, 1]]

    order = np.argsort(-pos_scores)
    positives = []

    for idx in order:
        y, x = pos_candidates[idx]
        ok = True
        for px, py in positives:
            if (x - px) ** 2 + (y - py) ** 2 < min_dist ** 2:
                ok = False
                break
        if ok:
            positives.append((int(x), int(y)))
        if len(positives) >= num_pos:
            break

    # 负点：在 bbox 外扩区域内，mask 外部采样
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    pad = max(30, int(max(box_w, box_h) * 0.35))

    ex0 = max(0, x0 - pad)
    ex1 = min(W - 1, x1 + pad)
    ey0 = max(0, y0 - pad)
    ey1 = min(H - 1, y1 + pad)

    region = np.zeros_like(mask_u8)
    region[ey0:ey1 + 1, ex0:ex1 + 1] = 1

    neg_region = (region > 0) & (mask_u8 == 0)

    inv = (mask_u8 == 0).astype(np.uint8)
    dist_to_mask = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    neg_candidates = np.column_stack(np.where(neg_region))
    negatives = []

    if len(neg_candidates) > 0:
        vals = dist_to_mask[neg_candidates[:, 0], neg_candidates[:, 1]]

        # 选离 mask 不太近也不太远的点
        target = np.percentile(vals, 35)
        order = np.argsort(np.abs(vals - target))

        for idx in order:
            y, x = neg_candidates[idx]
            ok = True
            for nx, ny in negatives:
                if (x - nx) ** 2 + (y - ny) ** 2 < min_dist ** 2:
                    ok = False
                    break
            if ok:
                negatives.append((int(x), int(y)))
            if len(negatives) >= num_neg:
                break

    return positives, negatives


def image_xy_to_page_xy(x, y, img_w, img_h, rect):
    """
    把原始图像像素坐标映射到网页中显示图像的位置。
    假设网页保持比例完整显示图像。
    """
    rect_x = rect["x"]
    rect_y = rect["y"]
    rect_w = rect["width"]
    rect_h = rect["height"]

    scale = min(rect_w / img_w, rect_h / img_h)

    disp_w = img_w * scale
    disp_h = img_h * scale

    offset_x = rect_x + (rect_w - disp_w) / 2.0
    offset_y = rect_y + (rect_h - disp_h) / 2.0

    page_x = offset_x + x * scale
    page_y = offset_y + y * scale

    return page_x, page_y


# ============================================================
# 网页元素查找 / 上传
# ============================================================

async def dump_page_debug(page, out_dir, name="debug_page"):
    """
    纯 SSH 下调试网页结构用：
      1. 保存当前页面 HTML
      2. 打印页面上可见按钮文本
    """
    ensure_dir(out_dir)

    html_path = os.path.join(out_dir, f"{name}.html")
    html = await page.content()

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Debug] saved html: {html_path}")

    try:
        texts = await page.locator("button").all_inner_texts()
        print("[Debug] buttons:")
        for i, t in enumerate(texts[:120]):
            t = t.strip().replace("\n", " ")
            if t:
                print(f"  button[{i}]: {t}")
    except Exception as e:
        print("[Debug] dump buttons failed:", repr(e))

    try:
        links = await page.locator("a").all_inner_texts()
        print("[Debug] links:")
        for i, t in enumerate(links[:80]):
            t = t.strip().replace("\n", " ")
            if t:
                print(f"  link[{i}]: {t}")
    except Exception as e:
        print("[Debug] dump links failed:", repr(e))


async def click_button_by_text(page, texts, timeout=5000):
    """
    根据文本点击按钮/元素。
    """
    for txt in texts:
        try:
            locator = page.get_by_text(txt, exact=False)
            await locator.first.click(timeout=timeout)
            print(f"[Click] text contains: {txt}")
            return True
        except Exception:
            pass
    return False


async def upload_image_via_filechooser(page, image_path, out_dir):
    """
    更鲁棒的上传方式：
      1. 如果 DOM 里已经有 input[type=file]，直接 set_input_files
      2. 否则监听 filechooser
      3. 自动点击 Upload / Choose / Select / Get started 等按钮
      4. 再不行就尝试点击所有 button
    """
    abs_img = os.path.abspath(image_path)

    # 方案 A：已有 file input
    inputs = await page.query_selector_all("input[type=file]")
    if inputs:
        print(f"[Upload] found input[type=file], count={len(inputs)}")
        await inputs[0].set_input_files(abs_img)
        return True

    await dump_page_debug(page, out_dir, name="01_no_file_input_dom")
    await save_screenshot(page, out_dir, "01_no_file_input.png")

    # 方案 B：点击会触发 file chooser 的按钮/文本
    upload_texts = [
        "Upload",
        "upload",
        "Upload image",
        "Upload Image",
        "Choose image",
        "Choose Image",
        "Choose file",
        "Choose File",
        "Select image",
        "Select Image",
        "Select file",
        "Select File",
        "Try it",
        "Try It",
        "Try Now",
        "Get started",
        "Get Started",
        "Start",
        "Open",
        "Browse",
        "开始",
        "上传",
        "选择图片",
        "选择文件",
        "浏览",
    ]

    for txt in upload_texts:
        try:
            print(f"[Upload] trying text: {txt}")
            locator = page.get_by_text(txt, exact=False).first

            async with page.expect_file_chooser(timeout=5000) as fc_info:
                await locator.click(timeout=3000)

            file_chooser = await fc_info.value
            await file_chooser.set_files(abs_img)
            print(f"[Upload] success via file chooser text: {txt}")
            return True

        except Exception:
            pass

    # 方案 C：点击所有 button，看哪个触发 file chooser
    try:
        buttons = await page.locator("button").all()
    except Exception:
        buttons = []

    print(f"[Upload] trying all buttons, count={len(buttons)}")

    for i, btn in enumerate(buttons):
        try:
            text = ""
            try:
                text = (await btn.inner_text()).strip().replace("\n", " ")
            except Exception:
                pass

            print(f"[Upload] trying button[{i}]: {text}")

            async with page.expect_file_chooser(timeout=4000) as fc_info:
                await btn.click(timeout=3000)

            file_chooser = await fc_info.value
            await file_chooser.set_files(abs_img)
            print(f"[Upload] success via button[{i}]: {text}")
            return True

        except Exception:
            continue

    # 方案 D：尝试点击页面上所有可见 input 附近/label
    try:
        labels = await page.locator("label").all()
    except Exception:
        labels = []

    print(f"[Upload] trying all labels, count={len(labels)}")

    for i, lab in enumerate(labels):
        try:
            text = ""
            try:
                text = (await lab.inner_text()).strip().replace("\n", " ")
            except Exception:
                pass

            print(f"[Upload] trying label[{i}]: {text}")

            async with page.expect_file_chooser(timeout=4000) as fc_info:
                await lab.click(timeout=3000)

            file_chooser = await fc_info.value
            await file_chooser.set_files(abs_img)
            print(f"[Upload] success via label[{i}]: {text}")
            return True

        except Exception:
            continue

    await save_screenshot(page, out_dir, "01_upload_failed.png")
    raise RuntimeError(
        "无法触发文件上传。请查看："
        f"{os.path.join(out_dir, '01_no_file_input_dom.html')} 和 01_upload_failed.png"
    )


async def get_main_image_rect(page):
    """
    获取网页左侧显示 RGB 图像的矩形区域。

    启发式：
      1. 找最大的 img
      2. 找最大的 canvas
      3. 找 role/img 或 visible element

    如果官网 DOM 变化，这里可能需要按截图调整。
    """
    best = None
    best_area = 0
    best_kind = None

    imgs = await page.query_selector_all("img")
    for img in imgs:
        try:
            box = await img.bounding_box()
        except Exception:
            box = None
        if not box:
            continue
        area = box["width"] * box["height"]
        if area > best_area:
            best = box
            best_area = area
            best_kind = "img"

    canvases = await page.query_selector_all("canvas")
    for c in canvases:
        try:
            box = await c.bounding_box()
        except Exception:
            box = None
        if not box:
            continue
        area = box["width"] * box["height"]
        if area > best_area:
            best = box
            best_area = area
            best_kind = "canvas"

    # 再找 svg，有些交互区域可能是 svg
    svgs = await page.query_selector_all("svg")
    for s in svgs:
        try:
            box = await s.bounding_box()
        except Exception:
            box = None
        if not box:
            continue
        area = box["width"] * box["height"]
        if area > best_area:
            best = box
            best_area = area
            best_kind = "svg"

    if best is not None and best_area > 10000:
        print(f"[Rect] selected {best_kind} rect: {best}")
        return best

    raise RuntimeError(
        "无法自动找到网页中的主图像区域。"
        "请查看 02_after_upload.png，根据实际 DOM 修改 get_main_image_rect()。"
    )


# ============================================================
# 点击 mask 点
# ============================================================

async def apply_mask_by_clicks(page, image_path, mask_path, num_pos=8, num_neg=8):
    img_w, img_h = load_image_size(image_path)
    mask = load_mask(mask_path)

    if mask.shape[:2] != (img_h, img_w):
        print(f"[Resize] mask {mask.shape[:2]} -> image {(img_h, img_w)}")
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    min_dist = max(12, min(img_w, img_h) // 30)

    positives, negatives = sample_prompt_points(
        mask,
        num_pos=num_pos,
        num_neg=num_neg,
        min_dist=min_dist,
    )

    print(f"[Prompt] positive points ({len(positives)}): {positives}")
    print(f"[Prompt] negative points ({len(negatives)}): {negatives}")

    rect = await get_main_image_rect(page)

    print("[Click] positive points")
    for x, y in positives:
        px, py = image_xy_to_page_xy(x, y, img_w, img_h, rect)
        await page.mouse.click(px, py)
        await page.wait_for_timeout(450)

    # 尝试切换负点 / 删除模式
    negative_mode_clicked = await click_button_by_text(
        page,
        [
            "Remove",
            "remove",
            "Negative",
            "negative",
            "Erase",
            "erase",
            "Subtract",
            "删除",
            "减去",
        ],
        timeout=1500,
    )

    if negative_mode_clicked:
        print("[Click] negative points")
        for x, y in negatives:
            px, py = image_xy_to_page_xy(x, y, img_w, img_h, rect)
            await page.mouse.click(px, py)
            await page.wait_for_timeout(350)
    else:
        print("[Warn] 没找到负点/删除模式按钮，跳过 negative points")

    return positives, negatives


# ============================================================
# 下载与转换
# ============================================================

def ensure_supported_download(path):
    ext = os.path.splitext(path)[1].lower()
    supported = [".ply", ".obj", ".glb", ".gltf", ".zip"]
    if ext not in supported:
        print(f"[Warn] 下载文件后缀 {ext} 不在常见支持列表 {supported}")
    return path


def convert_to_ply(input_path, out_ply):
    """
    将下载文件统一转换成 PLY。
    支持：
      .ply 直接复制
      .obj/.glb/.gltf 用 trimesh 转
      .zip 解压后找 ply/obj/glb/gltf
    """
    ensure_dir(os.path.dirname(out_ply))

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".ply":
        shutil.copy2(input_path, out_ply)
        print(f"[Convert] copy ply -> {out_ply}")
        return out_ply

    if ext == ".zip":
        unzip_dir = os.path.join(os.path.dirname(out_ply), "unzipped")
        ensure_dir(unzip_dir)

        print(f"[Unzip] {input_path} -> {unzip_dir}")
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(unzip_dir)

        candidates = []
        for root, _, files in os.walk(unzip_dir):
            for f in files:
                p = os.path.join(root, f)
                if os.path.splitext(p)[1].lower() in [".ply", ".obj", ".glb", ".gltf"]:
                    candidates.append(p)

        if not candidates:
            raise RuntimeError("zip 中没有找到 .ply/.obj/.glb/.gltf 文件")

        priority = {".ply": 0, ".obj": 1, ".glb": 2, ".gltf": 3}
        candidates.sort(key=lambda p: priority.get(os.path.splitext(p)[1].lower(), 99))

        print(f"[Unzip] selected model: {candidates[0]}")
        return convert_to_ply(candidates[0], out_ply)

    if ext in [".obj", ".glb", ".gltf"]:
        try:
            import trimesh
        except ImportError:
            raise RuntimeError("缺少 trimesh，请先运行：pip install trimesh")

        print(f"[Convert] {input_path} -> {out_ply}")

        loaded = trimesh.load(input_path, force=None)

        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) == 0:
                raise RuntimeError("trimesh.Scene 为空")
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        else:
            mesh = loaded

        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            raise RuntimeError(f"无法从 {input_path} 中读取有效 mesh")

        mesh.export(out_ply)
        print(f"[Convert] saved: {out_ply}")
        return out_ply

    raise RuntimeError(f"暂不支持转换格式: {ext}, file={input_path}")


# ============================================================
# 主自动化流程
# ============================================================

async def run_automation(args):
    from playwright.async_api import async_playwright

    ensure_dir(args.out_dir)
    ensure_dir(args.download_dir)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not args.headed,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--use-gl=swiftshader",
                "--enable-webgl",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--window-size=1800,1000",
            ],
        )

        context = await browser.new_context(
            accept_downloads=True,
            viewport={"width": args.viewport_w, "height": args.viewport_h},
            ignore_https_errors=True,
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )

        page = await context.new_page()

        print("[Open]", META_SAM3D_URL)

        # 纯 SSH 下不要等 domcontentloaded，官网前端资源可能很慢
        try:
            await page.goto(META_SAM3D_URL, wait_until="commit", timeout=60000)
            await page.wait_for_timeout(args.open_wait_ms)
            print("[Open] current url:", page.url)
            print("[Open] title:", await page.title())
            await save_screenshot(page, args.out_dir, "01_after_open.png")
        except Exception as e:
            print("[ERROR] 打开官网失败:", repr(e))
            await save_screenshot(page, args.out_dir, "01_open_failed.png")
            await context.close()
            await browser.close()
            raise

        # 上传 RGB
        print("[Upload RGB]", os.path.abspath(args.rgb))

        try:
            await upload_image_via_filechooser(page, args.rgb, args.out_dir)
        except Exception:
            await save_screenshot(page, args.out_dir, "01_upload_exception.png")
            await context.close()
            await browser.close()
            raise

        await page.wait_for_timeout(args.upload_wait_ms)
        await save_screenshot(page, args.out_dir, "02_after_upload.png")

        # 根据 mask 点击提示点
        try:
            await apply_mask_by_clicks(
                page,
                image_path=args.rgb,
                mask_path=args.mask,
                num_pos=args.num_pos,
                num_neg=args.num_neg,
            )
        except Exception:
            await save_screenshot(page, args.out_dir, "02_click_mask_exception.png")
            await context.close()
            await browser.close()
            raise

        await page.wait_for_timeout(3000)
        await save_screenshot(page, args.out_dir, "03_after_click_mask.png")

        # 点击生成 3D
        ok = await click_button_by_text(
            page,
            [
                "Generate 3D",
                "Generate",
                "Create 3D",
                "Convert to 3D",
                "生成3D模型",
                "生成 3D 模型",
                "生成",
                "3D",
            ],
            timeout=10000,
        )

        if not ok:
            await dump_page_debug(page, args.out_dir, name="03_no_generate_button_dom")
            await save_screenshot(page, args.out_dir, "03_no_generate_button.png")
            await context.close()
            await browser.close()
            raise RuntimeError(
                "没有找到 Generate/生成按钮。"
                "请查看 03_after_click_mask.png 和 03_no_generate_button.png。"
            )

        print("[Generate] clicked")
        await page.wait_for_timeout(5000)
        await save_screenshot(page, args.out_dir, "04_after_generate.png")

        # 等待生成完成并下载
        download_path = None
        deadline = time.time() + args.timeout_sec

        print("[Download] waiting for Download button / download event ...")

        while time.time() < deadline:
            try:
                async with page.expect_download(timeout=8000) as download_info:
                    clicked = await click_button_by_text(
                        page,
                        [
                            "Download",
                            "download",
                            "Export",
                            "export",
                            "Save",
                            "下载",
                            "导出",
                            "保存",
                        ],
                        timeout=5000,
                    )
                    if not clicked:
                        raise RuntimeError("download button not found")

                download = await download_info.value
                suggested = download.suggested_filename
                if not suggested:
                    suggested = "meta_sam3d_download.bin"

                save_path = os.path.join(args.download_dir, suggested)
                await download.save_as(save_path)

                download_path = save_path
                print("[Downloaded]", os.path.abspath(download_path))
                break

            except Exception as e:
                print("[Wait] 下载未就绪，继续等待...", repr(e))
                await page.wait_for_timeout(8000)

                # 周期性截图
                now = int(time.time())
                await save_screenshot(page, args.out_dir, f"04_waiting_download_{now}.png")

        if download_path is None:
            await dump_page_debug(page, args.out_dir, name="05_download_timeout_dom")
            await save_screenshot(page, args.out_dir, "05_download_timeout.png")
            await context.close()
            await browser.close()
            raise TimeoutError(
                f"等待下载超时: {args.timeout_sec}s。"
                "请查看 05_download_timeout.png。"
            )

        await context.close()
        await browser.close()

    return ensure_supported_download(download_path)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb", required=True, help="输入 RGB 图")
    parser.add_argument("--mask", required=True, help="输入 mask，脚本会按此 mask 采样点击点")

    parser.add_argument("--out-dir", default="./output_meta_web_auto")
    parser.add_argument("--download-dir", default="./meta_web_downloads")

    parser.add_argument("--num-pos", type=int, default=16)
    parser.add_argument("--num-neg", type=int, default=8)

    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--viewport-w", type=int, default=1800)
    parser.add_argument("--viewport-h", type=int, default=1000)

    parser.add_argument("--open-wait-ms", type=int, default=20000)
    parser.add_argument("--upload-wait-ms", type=int, default=12000)

    parser.add_argument("--headed", action="store_true", help="有图形界面时可用；纯 SSH 不要加")
    parser.add_argument("--no-convert", action="store_true", help="不转换成 PLY")

    args = parser.parse_args()

    if not os.path.exists(args.rgb):
        raise FileNotFoundError(f"找不到 RGB: {args.rgb}")

    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"找不到 mask: {args.mask}")

    ensure_dir(args.out_dir)
    ensure_dir(args.download_dir)

    print("=" * 80)
    print("[Start] Meta SAM3D 官网自动化")
    print("=" * 80)
    print(f"[URL]  {META_SAM3D_URL}")
    print(f"[RGB]  {args.rgb}")
    print(f"[Mask] {args.mask}")
    print(f"[Out]  {args.out_dir}")
    print(f"[Down] {args.download_dir}")
    print("=" * 80)

    downloaded = asyncio.run(run_automation(args))

    info = {
        "url": META_SAM3D_URL,
        "rgb": os.path.abspath(args.rgb),
        "mask": os.path.abspath(args.mask),
        "downloaded": os.path.abspath(downloaded),
        "converted_ply": None,
    }

    if not args.no_convert:
        out_ply = os.path.join(args.out_dir, "meta_web_model.ply")
        convert_to_ply(downloaded, out_ply)
        info["converted_ply"] = os.path.abspath(out_ply)
        print("[PLY]", os.path.abspath(out_ply))

    info_path = os.path.join(args.out_dir, "meta_web_auto_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n[Done]")
    print(f"  downloaded : {os.path.abspath(downloaded)}")
    if info["converted_ply"]:
        print(f"  ply        : {info['converted_ply']}")
    print(f"  info       : {os.path.abspath(info_path)}")
    print("\n✅ 自动官网生成流程完成")


if __name__ == "__main__":
    main()