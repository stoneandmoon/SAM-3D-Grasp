#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import numpy as np
import trimesh
import pyrender

# ---------------- meta / intrinsics ----------------
def load_meta(meta_path: str):
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def load_K_from_meta(meta: dict) -> np.ndarray:
    for k in ["camMat", "K", "cam_intr", "intrinsics", "camera_intrinsics"]:
        if isinstance(meta, dict) and k in meta:
            K = np.array(meta[k], dtype=np.float32)
            if K.shape == (3, 3):
                return K
    raise KeyError("meta 中未找到 3x3 相机内参（camMat/K 等）")

def load_vec3(meta: dict, key: str):
    if isinstance(meta, dict) and key in meta:
        v = np.array(meta[key], dtype=np.float32).reshape(-1)
        if v.shape[0] == 3:
            return v
    return None

# ---------------- geometry io ----------------
def load_mesh(ply_path: str) -> trimesh.Trimesh:
    g = trimesh.load(ply_path, process=False)
    if isinstance(g, trimesh.Scene):
        meshes = []
        for gg in g.geometry.values():
            if isinstance(gg, trimesh.Trimesh) and gg.faces is not None and len(gg.faces) > 0:
                meshes.append(gg)
        if len(meshes) == 0:
            raise ValueError(f"{ply_path} 不是 mesh（没有 faces）")
        return trimesh.util.concatenate(meshes)
    if isinstance(g, trimesh.Trimesh) and g.faces is not None and len(g.faces) > 0:
        return g
    raise ValueError(f"{ply_path} 不是 mesh（没有 faces）")

def scale_mesh(mesh: trimesh.Trimesh, s: float) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices = (np.asarray(m.vertices, dtype=np.float32) * float(s)).astype(np.float32)
    return m

# ---------------- coord transforms ----------------
def gl_to_cv_points(pts_gl: np.ndarray) -> np.ndarray:
    p = pts_gl.astype(np.float32).copy()
    p[:, 1] *= -1.0
    p[:, 2] *= -1.0
    return p

def cv_to_gl_mesh(mesh_cv: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh_cv.copy()
    v = np.asarray(m.vertices, dtype=np.float32)
    v[:, 1] *= -1.0
    v[:, 2] *= -1.0
    m.vertices = v
    return m

# ---------------- projection ----------------
def project_points_cv(K: np.ndarray, pts_cv: np.ndarray, W: int, H: int):
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    x, y, z = pts_cv[:, 0], pts_cv[:, 1], pts_cv[:, 2]
    valid = z > 1e-6
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return (np.zeros((0,), np.int32), np.zeros((0,), np.int32),
                np.zeros((0,), np.float32), np.zeros((0,), np.int64))

    x, y, z = x[valid], y[valid], z[valid]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if inb.sum() == 0:
        return (np.zeros((0,), np.int32), np.zeros((0,), np.int32),
                np.zeros((0,), np.float32), np.zeros((0,), np.int64))

    return ui[inb], vi[inb], z[inb].astype(np.float32), idx[inb].astype(np.int64)

def project_inbounds_count(K, verts_cv, W, H):
    u, v, z, idx = project_points_cv(K, verts_cv, W, H)
    return int(u.size)

def choose_best_hypothesis(K, hand_v0, obj_v0, W, H, meta_objTrans=None):
    scales = [1.0, 1e-3, 1e-2, 1e-1]
    best = None  

    for s in scales:
        hand_s = hand_v0 * float(s)
        obj_s  = obj_v0  * float(s)

        prior = 0.0
        if meta_objTrans is not None:
            c = obj_s.mean(axis=0)
            d_mesh = float(np.linalg.norm(c))
            d_meta = float(np.linalg.norm(meta_objTrans))
            if d_meta > 1e-6:
                prior = -abs(np.log((d_mesh + 1e-6) / (d_meta + 1e-6)))

        vertsA = np.vstack([gl_to_cv_points(hand_s), gl_to_cv_points(obj_s)]).astype(np.float32)
        scoreA = project_inbounds_count(K, vertsA, W, H) + 0.05 * prior
        if best is None or scoreA > best[0]:
            best = (scoreA, "OPENGL", s)

        vertsB = np.vstack([hand_s, obj_s]).astype(np.float32)
        scoreB = project_inbounds_count(K, vertsB, W, H) + 0.05 * prior
        if best is None or scoreB > best[0]:
            best = (scoreB, "OPENCV", s)

    return best  

# ---------------- rendering depth (strict visibility) ----------------
def estimate_znear_zfar_from_gl_meshes(meshes_gl):
    allv = np.vstack([np.asarray(m.vertices, dtype=np.float32) for m in meshes_gl]).astype(np.float32)
    d = (-allv[:, 2])  
    d = d[np.isfinite(d)]
    d = d[d > 1e-6]
    if d.size == 0:
        return 0.01, 10.0
    dmin = float(np.percentile(d, 1))
    dmax = float(np.percentile(d, 99))
    znear = max(1e-3, dmin * 0.5)
    zfar  = max(znear + 1e-2, dmax * 2.0)
    return znear, zfar

def render_depth_map_gl(K, meshes_gl, W, H):
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    znear, zfar = estimate_znear_zfar_from_gl_meshes(meshes_gl)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.25, 0.25, 0.25])

    for m in meshes_gl:
        pm = pyrender.Mesh.from_trimesh(m, smooth=False)
        scene.add(pm)

    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)
    cam_pose = np.eye(4, dtype=np.float32)  
    scene.add(cam, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    _, depth = r.render(scene)
    r.delete()

    return depth, znear, zfar

# ---------------- ply writer ----------------
def save_ply_xyz(path: str, pts: np.ndarray):
    n = pts.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = pts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--rgb-size", default="640x480")
    ap.add_argument("--hand-ply", required=True)
    ap.add_argument("--obj-ply", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--sample-hand", type=int, default=400000)
    ap.add_argument("--sample-obj", type=int, default=600000)

    ap.add_argument("--coord", choices=["AUTO", "OPENGL", "OPENCV"], default="AUTO")
    ap.add_argument("--scale", type=float, default=-1.0)

    ap.add_argument("--tol", type=float, default=0.002, help="深度一致性阈值(米)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        W, H = args.rgb_size.lower().split("x")
        W, H = int(W), int(H)
    except Exception:
        raise ValueError("--rgb-size 格式应为 WxH")

    np.random.seed(args.seed)

    meta = load_meta(args.meta)
    K = load_K_from_meta(meta)
    objT = load_vec3(meta, "objTrans")

    hand_mesh0 = load_mesh(args.hand_ply)
    obj_mesh0  = load_mesh(args.obj_ply)

    hand_v0 = np.asarray(hand_mesh0.vertices, dtype=np.float32)
    obj_v0  = np.asarray(obj_mesh0.vertices, dtype=np.float32)

    if args.coord != "AUTO" and args.scale > 0:
        chosen_coord = args.coord
        chosen_scale = float(args.scale)
        chosen_score = None
    else:
        score, coord, scale = choose_best_hypothesis(K, hand_v0, obj_v0, W, H, meta_objTrans=objT)
        chosen_coord = args.coord if args.coord != "AUTO" else coord
        chosen_scale = float(args.scale) if args.scale > 0 else float(scale)
        chosen_score = score

    hand_mesh_s = scale_mesh(hand_mesh0, chosen_scale)
    obj_mesh_s  = scale_mesh(obj_mesh0,  chosen_scale)

    if chosen_coord == "OPENGL":
        hand_gl = hand_mesh_s
        obj_gl  = obj_mesh_s
        used_T = "mesh already OPENGL"
    else:
        hand_gl = cv_to_gl_mesh(hand_mesh_s)
        obj_gl  = cv_to_gl_mesh(obj_mesh_s)
        used_T = "cv_to_gl (diag(1,-1,-1))"

    depth_map, znear, zfar = render_depth_map_gl(K, [hand_gl, obj_gl], W, H)

    hand_pts_gl, _ = trimesh.sample.sample_surface(hand_gl, args.sample_hand)
    obj_pts_gl,  _ = trimesh.sample.sample_surface(obj_gl,  args.sample_obj)
    hand_pts_gl = hand_pts_gl.astype(np.float32)
    obj_pts_gl  = obj_pts_gl.astype(np.float32)

    hand_pts_cv = gl_to_cv_points(hand_pts_gl)
    obj_pts_cv  = gl_to_cv_points(obj_pts_gl)

    hu, hv, hz, hidx = project_points_cv(K, hand_pts_cv, W, H)
    ou, ov, oz, oidx = project_points_cv(K, obj_pts_cv,  W, H)

    tol = float(args.tol)

    def strict_keep(u, v, z, pts_cv):
        if u.size == 0:
            return np.zeros((0, 3), np.float32)
        d = depth_map[v, u].astype(np.float32)
        valid = d > 0
        if valid.sum() == 0:
            return np.zeros((0, 3), np.float32)
        keep = valid & (np.abs(z - d) <= tol)
        if keep.sum() == 0:
            return np.zeros((0, 3), np.float32)
        return pts_cv[keep].astype(np.float32)

    hand_pts_cv_proj = hand_pts_cv[hidx] if hidx.size else np.zeros((0, 3), np.float32)
    obj_pts_cv_proj  = obj_pts_cv[oidx]  if oidx.size else np.zeros((0, 3), np.float32)

    hand_vis_cv = strict_keep(hu, hv, hz, hand_pts_cv_proj)
    obj_vis_cv  = strict_keep(ou, ov, oz, obj_pts_cv_proj)

    all_vis_cv = np.vstack([hand_vis_cv, obj_vis_cv]) if (hand_vis_cv.size or obj_vis_cv.size) else np.zeros((0, 3), np.float32)

    save_ply_xyz(os.path.join(args.out_dir, "visible_hand_cv_strict.ply"), hand_vis_cv)
    save_ply_xyz(os.path.join(args.out_dir, "visible_obj_cv_strict.ply"),  obj_vis_cv)
    save_ply_xyz(os.path.join(args.out_dir, "visible_all_cv_strict.ply"),  all_vis_cv)

    np.savez(
        os.path.join(args.out_dir, "visible_strict_cv.npz"),
        hand_vis_cv=hand_vis_cv,
        obj_vis_cv=obj_vis_cv,
        all_vis_cv=all_vis_cv,
        chosen_coord=chosen_coord,
        chosen_scale=chosen_scale,
        tol=tol
    )

    with open(os.path.join(args.out_dir, "debug_strict.txt"), "w", encoding="utf-8") as f:
        f.write(f"rgb_size={W}x{H}\n")
        f.write(f"K=\n{K}\n")
        if objT is not None:
            f.write(f"meta_objTrans={objT.tolist()} norm={float(np.linalg.norm(objT))}\n")
        f.write(f"chosen_coord={chosen_coord}\n")
        f.write(f"chosen_scale={chosen_scale}\n")
        f.write(f"used_T={used_T}\n")
        if chosen_score is not None:
            f.write(f"auto_score={chosen_score}\n")
        f.write(f"znear={znear} zfar={zfar}\n")
        f.write(f"depth_valid_pixels={int((depth_map>0).sum())}\n")
        f.write(f"proj_hand={int(hu.size)} proj_obj={int(ou.size)}\n")
        f.write(f"strict_visible_hand={int(hand_vis_cv.shape[0])} strict_visible_obj={int(obj_vis_cv.shape[0])} strict_visible_all={int(all_vis_cv.shape[0])}\n")
        f.write(f"tol={tol}\n")

    print("[OK] strict visible points saved (CV camera coords):")
    print(" - visible_all_cv_strict.ply")
    print(" - visible_hand_cv_strict.ply")
    print(" - visible_obj_cv_strict.ply")
    print(" - debug_strict.txt")

if __name__ == "__main__":
    main()
