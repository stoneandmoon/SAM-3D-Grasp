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

    mesh_raw = trimesh.load(find_mesh(args.model_dir), process=False, force='mesh')
    diag = np.linalg.norm(mesh_raw.bounding_box.extents)
    if diag > 1.0: mesh_raw.apply_scale(0.001)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    mesh_raw.apply_transform(T)

    # ==========================================
    # 核心修复：坐标系自动侦测与分发
    # ==========================================
    z_mean = mesh_raw.vertices[:, 2].mean()
    if z_mean < 0:
        print("[检测] 发现原生坐标系为 OpenGL (Z < 0)。启动自适应翻转...")
        mesh_gl = mesh_raw.copy()
        mesh_cv = mesh_raw.copy()
        v = np.asarray(mesh_cv.vertices)
        v[:, 1] *= -1.0
        v[:, 2] *= -1.0
        mesh_cv.vertices = v
    else:
        print("[检测] 发现原生坐标系为 OpenCV (Z > 0)。启动自适应翻转...")
        mesh_cv = mesh_raw.copy()
        mesh_gl = mesh_raw.copy()
        v = np.asarray(mesh_gl.vertices)
        v[:, 1] *= -1.0
        v[:, 2] *= -1.0
        mesh_gl.vertices = v

    print("[渲染] 生成虚拟深度图以剔除遮挡...")
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.5,0.5,0.5])
    scene.add(pyrender.Mesh.from_trimesh(mesh_gl, smooth=False))
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    # 注意 PyRender 的画面 Y 轴翻转
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=H-cy, znear=0.01, zfar=10.0)
    scene.add(cam, pose=np.eye(4))
    
    r = pyrender.OffscreenRenderer(W, H)
    depth_map = r.render(scene)[1]
    r.delete()

    print("[提取] 计算相交并生成点云...")
    pts_cv, _ = trimesh.sample.sample_surface(mesh_cv, 200000)
    x, y, z = pts_cv[:,0], pts_cv[:,1], pts_cv[:,2]
    
    # 在 OpenCV 坐标系下，Z 必定是正数
    valid = z > 1e-6
    pts_cv, x, y, z = pts_cv[valid], x[valid], y[valid], z[valid]
    
    u = np.round(fx * (x / z) + cx).astype(np.int32)
    v = np.round(fy * (y / z) + cy).astype(np.int32)
    
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts_cv, u, v, z = pts_cv[inb], u[inb], v[inb], z[inb]
    
    # 遮挡测试
    d = depth_map[v, u]
    keep = (d > 0) & (np.abs(z - d) <= 0.01)
    vis_pts = pts_cv[keep]

    out_file = os.path.join(args.out_dir, "visible_all_cv_strict.ply")
    with open(out_file, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nend_header\n".format(len(vis_pts)))
        for pt in vis_pts: f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
            
    print(f"✅ 提取大成功！共获得 {len(vis_pts)} 个残缺点。")
    print(f"📁 已保存至: {out_file}")

if __name__ == "__main__":
    main()
