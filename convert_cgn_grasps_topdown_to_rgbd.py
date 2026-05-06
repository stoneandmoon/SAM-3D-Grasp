#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np


def unwrap(x):
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        return x.item()
    return x


def is_valid_grasp(g):
    g = np.asarray(g)
    return g.shape == (4, 4) and np.all(np.isfinite(g)) and not np.allclose(g, -1.0)


def transform_grasps(pred_grasps_top, T_top_to_rgbd):
    pred_grasps_top = unwrap(pred_grasps_top)

    if isinstance(pred_grasps_top, dict):
        out = {}
        for obj_id, grasps in pred_grasps_top.items():
            new_grasps = []
            for g in grasps:
                g = np.asarray(g, dtype=np.float64)
                if is_valid_grasp(g):
                    new_grasps.append(T_top_to_rgbd @ g)
                else:
                    new_grasps.append(g)
            out[obj_id] = np.asarray(new_grasps, dtype=np.float64)
        return out

    out = []
    for g in np.asarray(pred_grasps_top):
        g = np.asarray(g, dtype=np.float64)
        if is_valid_grasp(g):
            out.append(T_top_to_rgbd @ g)
        else:
            out.append(g)
    return np.asarray(out, dtype=np.float64)


def transform_point(p, T):
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return T[:3, :3] @ p + T[:3, 3]


def transform_contact_pts(contact_pts_top, T_top_to_rgbd):
    if contact_pts_top is None:
        return None

    contact_pts_top = unwrap(contact_pts_top)

    if isinstance(contact_pts_top, dict):
        out = {}
        for obj_id, pts in contact_pts_top.items():
            pts = np.asarray(pts)
            new_pts = []
            for p in pts:
                p = np.asarray(p)
                if p.shape[-1] == 3 and np.all(np.isfinite(p)):
                    new_pts.append(transform_point(p, T_top_to_rgbd))
                else:
                    new_pts.append(p)
            out[obj_id] = np.asarray(new_pts)
        return out

    pts = np.asarray(contact_pts_top)
    if pts.ndim >= 2 and pts.shape[-1] == 3:
        flat = pts.reshape(-1, 3)
        flat_new = np.asarray([transform_point(p, T_top_to_rgbd) for p in flat])
        return flat_new.reshape(pts.shape)

    return contact_pts_top


def collect_records(pred_grasps_rgbd, scores, contact_pts_rgbd=None):
    pred_grasps_rgbd = unwrap(pred_grasps_rgbd)
    scores = unwrap(scores)
    contact_pts_rgbd = unwrap(contact_pts_rgbd) if contact_pts_rgbd is not None else None

    records = []

    if isinstance(pred_grasps_rgbd, dict):
        for obj_id, grasps in pred_grasps_rgbd.items():
            if isinstance(scores, dict):
                obj_scores = scores[obj_id]
            else:
                obj_scores = scores

            obj_contacts = None
            if isinstance(contact_pts_rgbd, dict) and obj_id in contact_pts_rgbd:
                obj_contacts = contact_pts_rgbd[obj_id]

            for i, g in enumerate(grasps):
                if not is_valid_grasp(g):
                    continue

                score = float(obj_scores[i]) if i < len(obj_scores) else 0.0

                rec = {
                    "obj_id": str(obj_id),
                    "index": int(i),
                    "score": score,
                    "T_grasp_rgbd": np.asarray(g, dtype=float).tolist(),
                    "position_rgbd": np.asarray(g[:3, 3], dtype=float).tolist(),
                    "rotation_rgbd": np.asarray(g[:3, :3], dtype=float).tolist(),
                }

                if obj_contacts is not None and i < len(obj_contacts):
                    cp = np.asarray(obj_contacts[i])
                    if cp.shape[-1] == 3:
                        rec["contact_point_rgbd"] = cp.astype(float).tolist()

                records.append(rec)

    else:
        grasps = np.asarray(pred_grasps_rgbd)
        obj_scores = np.asarray(scores)
        contacts = np.asarray(contact_pts_rgbd) if contact_pts_rgbd is not None else None

        for i, g in enumerate(grasps):
            if not is_valid_grasp(g):
                continue

            score = float(obj_scores[i]) if i < len(obj_scores) else 0.0

            rec = {
                "obj_id": "0",
                "index": int(i),
                "score": score,
                "T_grasp_rgbd": np.asarray(g, dtype=float).tolist(),
                "position_rgbd": np.asarray(g[:3, 3], dtype=float).tolist(),
                "rotation_rgbd": np.asarray(g[:3, :3], dtype=float).tolist(),
            }

            if contacts is not None and i < len(contacts):
                cp = np.asarray(contacts[i])
                if cp.shape[-1] == 3:
                    rec["contact_point_rgbd"] = cp.astype(float).tolist()

            records.append(rec)

    records.sort(key=lambda x: x["score"], reverse=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-npz", required=True)
    parser.add_argument("--topdown-json", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.pred_npz):
        raise FileNotFoundError(args.pred_npz)
    if not os.path.exists(args.topdown_json):
        raise FileNotFoundError(args.topdown_json)

    with open(args.topdown_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "T_topdown_to_rgbd" not in meta:
        raise RuntimeError("topdown json 里找不到 T_topdown_to_rgbd")

    T_top_to_rgbd = np.asarray(meta["T_topdown_to_rgbd"], dtype=np.float64)

    data = np.load(args.pred_npz, allow_pickle=True)

    pred_grasps_top = data["pred_grasps_cam"]
    scores = data["scores"]
    contact_pts_top = data["contact_pts"] if "contact_pts" in data.files else None

    pred_grasps_rgbd = transform_grasps(pred_grasps_top, T_top_to_rgbd)
    contact_pts_rgbd = transform_contact_pts(contact_pts_top, T_top_to_rgbd)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)

    np.savez(
        args.out_npz,
        pred_grasps_cam=pred_grasps_rgbd,
        scores=scores,
        contact_pts=contact_pts_rgbd,
    )

    records = collect_records(pred_grasps_rgbd, scores, contact_pts_rgbd)
    top = records[:args.top_k]

    out_obj = {
        "source_prediction_npz": os.path.abspath(args.pred_npz),
        "topdown_json": os.path.abspath(args.topdown_json),
        "coordinate_frame": "RGB-D camera frame",
        "definition": "T_grasp_rgbd = T_topdown_to_rgbd @ T_grasp_topdown",
        "num_valid_grasps": len(records),
        "top_k": len(top),
        "grasps": top,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print("\n✅ 已把 Contact-GraspNet 抓取结果从 top-down frame 转回 RGB-D frame")
    print(f"输入 NPZ : {os.path.abspath(args.pred_npz)}")
    print(f"输出 NPZ : {os.path.abspath(args.out_npz)}")
    print(f"输出 JSON: {os.path.abspath(args.out_json)}")
    print(f"valid grasps: {len(records)}")

    print("\nTop grasps:")
    for i, r in enumerate(top[:10]):
        print(
            f"  #{i:02d} "
            f"score={r['score']:.4f}, "
            f"index={r['index']}, "
            f"pos={r['position_rgbd']}"
        )


if __name__ == "__main__":
    main()
