#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, pickle, glob, cv2
import numpy as np
import trimesh
import pyrender

def load_meta(meta_path):
    with open(meta_path, "rb") as f: return pickle.load(f)

def load_K_from_meta(meta):
    for k in ["camMat", "K", "cam_intr"]:
        if k in meta: return np.array(meta[k], dtype=np.float32)
    return np.eye(3, dtype=np.float32)

def objrot_to_R(objRot):
    r = np.array(objRot, dtype=np.float32).reshape(-1)
    if r.size == 9: return r.reshape(3, 3)
    if r.size == 3: R, _ = cv2.Rodrigues(r.reshape(3, 1)); return R
    return np.eye(3, dtype=np.float32)

def find_mesh(model_dir):
    cands = glob.glob(os.path.join(model_dir, "*.ply")) + glob.glob(os.path.join(model_dir, "*.obj"))
    for c in cands:
        if "textured_simple" in c: return c
    return cands[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rgb-size", default="640x480")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    W, H = [int(x) for x in args.rgb_size.split("x")]

    meta = load_meta(args.meta)
    K = load_K_from_meta(meta)
    R = objrot_to_R(meta["objRot"])
    t = np.array(meta["objTrans"], dtype=np.float32).reshape(3)

    print("\n" + "="*50)
    print("🔍 X-Ray 诊断报告开始")
    print("="*50)
    print(f"[1] 相机内参 (K):\n{K}")
    print(f"[2] 物体位移 (t): {t} (单位如果是毫米会导致物体飞出视野！)")

    mesh_cv = trimesh.load(find_mesh(args.model_dir), process=False, force='mesh')
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    mesh_cv.apply_transform(T)

    z_vals = mesh_cv.vertices[:, 2]
    print(f"[3] 变换后物体的 Z 轴深度范围: Min={z_vals.min():.4f}, Max={z_vals.max():.4f}")
    if z_vals.max() < 0:
        print("❌ 致命错误：物体在相机的背后 (Z < 0)！")

    # 转为 OpenGL 坐标系供 PyRender 使用
    mesh_gl = mesh_cv.copy()
    v = np.asarray(mesh_gl.vertices)
    v[:, 1] *= -1.0
    v[:, 2] *= -1.0
    mesh_gl.vertices = v

    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh_gl, smooth=False))
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    # 【核心修复猜想】：PyRender 的 Y 轴原点在左下角，而 OpenCV 在左上角！
    # 如果 cy 不修正，投影会发生上下错位，导致点云和深度图完美错过！
    cy_gl = H - cy 
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy_gl, znear=0.01, zfar=10.0)
    scene.add(cam, pose=np.eye(4))
    
    r = pyrender.OffscreenRenderer(W, H)
    color, depth_map = r.render(scene)
    r.delete()

    valid_depths = depth_map[depth_map > 0]
    print(f"[4] PyRender 渲染的深度图: 有效像素点数 = {valid_depths.size}")
    if valid_depths.size == 0:
        print("❌ 致命错误：PyRender 渲染出了一张纯黑的图！可能超出 zfar/znear 范围。")
    else:
        print(f"    -> 渲染的深度范围: Min={valid_depths.min():.4f}, Max={valid_depths.max():.4f}")

    pts_cv, _ = trimesh.sample.sample_surface(mesh_cv, 100000)
    x, y, z = pts_cv[:,0], pts_cv[:,1], pts_cv[:,2]
    valid = z > 1e-6
    pts_cv, x, y, z = pts_cv[valid], x[valid], y[valid], z[valid]

    u = np.round(fx * (x / z) + cx).astype(np.int32)
    v = np.round(fy * (y / z) + cy).astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts_cv, u, v, z = pts_cv[inb], u[inb], v[inb], z[inb]
    
    print(f"[5] 投射到 2D 画面内的点数: {len(pts_cv)}")

    if len(pts_cv) > 0:
        d = depth_map[v, u]
        diff = np.abs(z - d)
        print(f"[6] 投影点与渲染深度的平均误差: {diff.mean():.6f} 米")
        
        # 放宽至 3 厘米容差进行极限测试
        keep = (d > 0) & (diff <= 0.03)
        vis_pts = pts_cv[keep]
        print(f"[7] 最终匹配成功的点数 (容差 3cm): {len(vis_pts)}")
    else:
        print("[6] 无法匹配，投射点数为 0。")

    # --- 生成诊断图像 ---
    diag_img = (depth_map > 0).astype(np.uint8) * 100  # 灰色背景代表深度图
    diag_img = cv2.cvtColor(diag_img, cv2.COLOR_GRAY2BGR)
    
    # 用红点画出数学投影算出来的位置
    for px, py in zip(u[::50], v[::50]): 
        cv2.circle(diag_img, (px, py), 1, (0, 0, 255), -1)
        
    cv2.imwrite(os.path.join(args.out_dir, "xray_diagnosis.jpg"), diag_img)
    print("="*50)
    print("📸 诊断图 xray_diagnosis.jpg 已生成！(红点为投影坐标，灰色为渲染遮罩)")

if __name__ == "__main__":
    main()
