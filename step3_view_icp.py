import os
import argparse
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import cv2
import matplotlib.pyplot as plt  # 修复plt导入问题

# ---------------------------------------------------------
# 1. 调试工具 - 显示点云范围
# ---------------------------------------------------------
def debug_point_cloud_range(name, points):
    """显示点云范围，帮助诊断坐标系问题"""
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    center = (min_pt + max_pt) / 2
    extent = max_pt - min_pt
    diag = np.linalg.norm(extent)
    
    print(f"    🔍 {name} 范围:")
    print(f"       中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"       尺寸: [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
    print(f"       对角线: {diag:.3f}m")
    return center, extent

# ---------------------------------------------------------
# 2. 优化版 SVD 配准 (增强稳定性)
# ---------------------------------------------------------
def svd_align(A, B):
    """稳定版SVD配准 - 处理极端情况"""
    if len(A) < 3 or len(B) < 3:
        print("⚠️ 点数不足，返回单位变换")
        return np.eye(3), np.zeros(3)
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    try:
        H = A_centered.T @ B_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        t = centroid_B - R @ centroid_A
        return R, t
    except np.linalg.LinAlgError:
        print("⚠️ SVD分解失败，返回单位变换")
        return np.eye(3), np.zeros(3)

def numpy_icp(source_pts, target_pts, T_init, max_iter=60, y_axis_weight=1.5):
    """稳健ICP - 增加异常处理"""
    src = np.copy(source_pts)
    tree = KDTree(target_pts)
    
    # 应用初始变换
    R_total = T_init[:3, :3]
    t_total = T_init[:3, 3]
    curr_pts = (src @ R_total.T) + t_total
    
    prev_rmse = float('inf')
    for i in range(max_iter):
        try:
            dist, idx = tree.query(curr_pts)
            
            # 动态内点过滤 (更稳健)
            max_dist = np.percentile(dist, 95)  # 使用95%分位数作为上限
            mask = dist < max_dist
            
            if np.sum(mask) < 50: 
                print(f"    ⚠️ 迭代{i}内点不足，提前终止")
                break
                
            # 重点：只使用有效点进行配准
            A_in = curr_pts[mask]
            B_in = target_pts[idx[mask]]
            
            # 避免退化情况
            if len(A_in) < 10 or len(B_in) < 10:
                break
                
            R_step, t_step = svd_align(A_in, B_in)
            
            # 应用变换
            curr_pts = (curr_pts @ R_step.T) + t_step
            R_total = R_step @ R_total
            t_total = R_step @ t_total + t_step
            
            # 收敛检查
            rmse = np.sqrt(np.mean(dist[mask]**2))
            if abs(prev_rmse - rmse) < 1e-6 or rmse < 0.001:
                break
            prev_rmse = rmse
            
        except Exception as e:
            print(f"    ⚠️ ICP迭代{i}失败: {str(e)}")
            break
    
    # 计算Fitness (使用3cm阈值)
    final_dist, _ = tree.query(curr_pts)
    fitness = np.sum(final_dist < 0.03) / len(final_dist)
    
    # Y轴对齐度
    y_axis_alignment = np.std(curr_pts[:, 1])
    
    return R_total, t_total, fitness, np.mean(final_dist), y_axis_alignment

# ---------------------------------------------------------
# 3. HO3D专用坐标系校正 (增强版)
# ---------------------------------------------------------
def ho3d_coordinate_correction(model_pts, partial_pts):
    """增强版HO3D坐标系校正 - 基于实际数据范围"""
    print("    🔧 执行增强坐标系校正...")
    
    # 1. 先显示原始范围
    debug_point_cloud_range("原始模型点云", model_pts)
    debug_point_cloud_range("原始部分点云", partial_pts)
    
    # 2. 关键修复：将点云移到原点附近
    model_center = np.mean(model_pts, axis=0)
    partial_center = np.mean(partial_pts, axis=0)
    
    model_pts = model_pts - model_center
    partial_pts = partial_pts - partial_center
    
    # 3. 分析主要方向
    model_std = np.std(model_pts, axis=0)
    partial_std = np.std(partial_pts, axis=0)
    
    print(f"    📊 模型标准差: X={model_std[0]:.3f}, Y={model_std[1]:.3f}, Z={model_std[2]:.3f}")
    print(f"    📊 部分标准差: X={partial_std[0]:.3f}, Y={partial_std[1]:.3f}, Z={partial_std[2]:.3f}")
    
    # 4. 智能轴交换 (基于实际数据)
    # 找出最大标准差的轴 (应该是高度方向)
    model_max_axis = np.argmax(model_std)
    partial_max_axis = np.argmax(partial_std)
    
    print(f"    🎯 模型最大变化轴: {['X','Y','Z'][model_max_axis]}")
    print(f"    🎯 部分最大变化轴: {['X','Y','Z'][partial_max_axis]}")
    
    # 5. 强制HO3D标准：Y轴为高度方向
    if model_max_axis != 1:  # 如果不是Y轴
        print(f"    🔀 交换轴: 将{['X','Y','Z'][model_max_axis]}轴映射到Y轴")
        
        # 创建新的点云顺序 [X, Y, Z] -> 根据最大轴调整
        if model_max_axis == 0:  # X轴是高度
            model_pts = model_pts[:, [1, 0, 2]]  # X,Y,Z -> Y,X,Z
            partial_pts = partial_pts[:, [1, 0, 2]]
        elif model_max_axis == 2:  # Z轴是高度
            model_pts = model_pts[:, [0, 2, 1]]  # X,Y,Z -> X,Z,Y
            partial_pts = partial_pts[:, [0, 2, 1]]
    
    # 6. 确保Y轴向上
    if np.mean(model_pts[:, 1]) < 0:
        print("    ⬆️ 翻转Y轴 (确保向上)")
        model_pts[:, 1] = -model_pts[:, 1]
        partial_pts[:, 1] = -partial_pts[:, 1]
    
    # 7. 显示校正后范围
    debug_point_cloud_range("校正后模型点云", model_pts)
    debug_point_cloud_range("校正后部分点云", partial_pts)
    
    return model_pts, partial_pts

def objrot_to_R(objRot):
    """安全版位姿转换"""
    try:
        r = np.array(objRot).reshape(-1)
        if r.size == 9: 
            return r.reshape(3, 3)
        R, _ = cv2.Rodrigues(r.reshape(3, 1))
        return R
    except:
        print("⚠️ 位姿转换失败，返回单位矩阵")
        return np.eye(3)

# ---------------------------------------------------------
# 4. 主流程 (终极修复版)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--sam3d", required=True)
    parser.add_argument("--partial", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载数据
    print("=========================================")
    print("[1] 正在加载点云与 Meta...")
    print("=========================================")
    sam_pcd = o3d.io.read_point_cloud(args.sam3d)
    part_pcd = o3d.io.read_point_cloud(args.partial)
    sam_pts = np.asarray(sam_pcd.points).astype(np.float64)
    part_pts = np.asarray(part_pcd.points).astype(np.float64)
    
    print(f"    ✓ SAM3D点云: {len(sam_pts)} 点")
    print(f"    ✓ Partial点云: {len(part_pts)} 点")
    
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
        if isinstance(meta, list): 
            meta = meta[0]
            print("    ✓ Meta文件是列表格式，已取第一个元素")
    
    print(f"    ✓ Meta内容: {list(meta.keys())}")
    if "objTrans" in meta:
        print(f"       物体位置: {meta['objTrans']}")
    if "objRot" in meta:
        print(f"       物体旋转: {meta['objRot'][:3] if isinstance(meta['objRot'], list) else meta['objRot']}")

    # 2. HO3D专用坐标系校正 (增强版)
    print("=========================================")
    print("[2] 正在执行增强HO3D坐标系校正...")
    print("=========================================")
    sam_pts, part_pts = ho3d_coordinate_correction(sam_pts, part_pts)

    # 3. 尺度恢复 (更稳健的版本)
    print("=========================================")
    print("[3] 正在恢复真实尺度...")
    print("=========================================")
    
    try:
        mesh = o3d.io.read_triangle_mesh(os.path.join(args.model_dir, "textured_simple.obj"))
        gt_diag = np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent())
        sam_diag = np.linalg.norm(np.max(sam_pts, 0) - np.min(sam_pts, 0))
        
        if sam_diag < 1e-6:
            print("⚠️ SAM3D点云对角线太小，使用默认尺度1.0")
            scale = 1.0
        else:
            scale = gt_diag / sam_diag
            print(f"    📏 GT对角线: {gt_diag:.4f}m")
            print(f"    📏 SAM3D对角线: {sam_diag:.4f}m")
        
        sam_pts = (sam_pts - np.mean(sam_pts, axis=0)) * scale
        print(f"    ✓ 尺度对齐完成: Scale={scale:.4f}")
    except Exception as e:
        print(f"⚠️ 尺度恢复失败: {str(e)}，使用默认尺度")
        scale = 1.0

    # 4. 构造优化初始方案 (增强鲁棒性)
    print("=========================================")
    print("[4] 正在测试最优初始方向...")
    print("=========================================")
    candidates = []
    
    # 方案 A: 质心直接对齐 (增强版)
    T_centroid = np.eye(4)
    T_centroid[:3, 3] = np.mean(part_pts, axis=0) - np.mean(sam_pts, axis=0)
    candidates.append(("Centroid", T_centroid))
    
    # 方案 B: Meta位姿 (带异常处理)
    if "objRot" in meta and "objTrans" in meta:
        try:
            R_m = objrot_to_R(meta["objRot"])
            t_m = np.array(meta["objTrans"]).reshape(3)
            
            # 确保t_m是3D向量
            if len(t_m) > 3:
                t_m = t_m[:3]
            
            T_meta = np.eye(4)
            T_meta[:3, :3] = R_m
            T_meta[:3, 3] = t_m
            candidates.append(("Meta_GT", T_meta))
            
            # 方案 C: Meta Y轴翻转
            R_flip = np.array([[1,0,0],[0,-1,0],[0,0,1]]) @ R_m
            T_flip = np.eye(4)
            T_flip[:3, :3] = R_flip
            T_flip[:3, 3] = t_m
            candidates.append(("Meta_YFlip", T_flip))
            
            # 方案 D: 单位矩阵 (零位移)
            T_identity = np.eye(4)
            candidates.append(("Identity", T_identity))
            
        except Exception as e:
            print(f"⚠️ Meta位姿处理失败: {str(e)}")
    
    # 如果没有Meta方案，添加基本方案
    if len(candidates) == 0:
        print("⚠️ 无有效Meta方案，使用基本方案")
        T_identity = np.eye(4)
        candidates.append(("Identity", T_identity))
        T_zero = np.eye(4)
        T_zero[:3, 3] = np.zeros(3)
        candidates.append(("Zero_Trans", T_zero))

    # 5. 跑分筛选 (使用降采样点云)
    print("=========================================")
    print(f"[5] 测试 {len(candidates)} 个候选方案...")
    print("=========================================")
    
    best_res = None
    best_fit = -1
    best_y_alignment = float('inf')
    
    # 降采样用于快速测试
    max_test_points = 1000
    sam_sub_idx = np.random.choice(len(sam_pts), min(max_test_points, len(sam_pts)), replace=False)
    part_sub_idx = np.random.choice(len(part_pts), min(2000, len(part_pts)), replace=False)
    sam_sub = sam_pts[sam_sub_idx]
    part_sub = part_pts[part_sub_idx]
    
    for name, T_init in candidates:
        try:
            R, t, fit, err, y_align = numpy_icp(
                sam_sub, part_sub, T_init, 
                max_iter=30, 
                y_axis_weight=1.5
            )
            print(f"    ✓ 方案 [{name:12s}] | Fitness: {fit:.4f} | RMSE: {err:.4f}m | Y-align: {y_align:.4f}")
            
            # 选择标准：优先Fitness，其次Y轴对齐
            if fit > best_fit or (abs(fit - best_fit) < 1e-4 and y_align < best_y_alignment):
                best_fit = fit
                best_y_alignment = y_align
                best_res = (R, t, name, T_init)
                
        except Exception as e:
            print(f"    ⚠️ 方案 [{name:12s}] 失败: {str(e)}")

    if best_res is None:
        print("⚠️ 所有方案失败，使用默认单位变换")
        final_R = np.eye(3)
        final_t = np.zeros(3)
        win_name = "Default"
    else:
        final_R, final_t, win_name, best_T_init = best_res
        print(f"=========================================")
        print(f"[6] 最终胜出方案: {win_name} (Fitness: {best_fit:.4f}, Y-align: {best_y_alignment:.4f})")
        print("=========================================")
        
        # 7. 使用最佳初始变换进行最终配准
        print("[7] 执行最终高精度配准 (使用完整点云)...")
        R_final, t_final, fit_final, err_final, y_align_final = numpy_icp(
            sam_pts, part_pts, best_T_init,
            max_iter=50,
            y_axis_weight=2.0
        )
        final_R, final_t = R_final, t_final
        print(f"    ✓ 最终配准结果 | Fitness: {fit_final:.4f} | RMSE: {err_final:.4f}m | Y-align: {y_align_final:.4f}")

    # 8. 生成最终结果
    sam_aligned = (sam_pts @ final_R.T) + final_t
    
    merged_pcd = o3d.geometry.PointCloud()
    merged_points = np.vstack([sam_aligned, part_pts])
    merged_colors = np.vstack([
        np.tile([0.1, 0.6, 1.0], (len(sam_aligned), 1)),  # 蓝色: 模型
        np.tile([1.0, 0.2, 0.2], (len(part_pts), 1))      # 红色: 部分点云
    ])
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    out_path = os.path.join(args.out_dir, "vis_fusion.ply")
    o3d.io.write_point_cloud(out_path, merged_pcd)
    
    # 9. 详细评估
    tree = KDTree(part_pts)
    dists, _ = tree.query(sam_aligned)
    y_axis_err = np.abs(sam_aligned[:, 1] - np.mean(sam_aligned[:, 1]))
    
    print("\n" + "="*50)
    print(f"✅ HO3D配准完成! (增强修复版)")
    print("="*50)
    print(f"平均距离: {np.mean(dists):.4f}m")
    print(f"中位数距离: {np.median(dists):.4f}m")
    print(f"Y轴对齐误差: {np.mean(y_axis_err):.4f}m")
    print(f"95%分位距离: {np.percentile(dists, 95):.4f}m")
    print(f"最佳方案: {win_name}")
    print(f"输出路径: {out_path}")
    print("="*50)
    
    # 10. 生成诊断可视化 (现在有plt了)
    print("[8] 生成诊断可视化...")
    vis_path = os.path.join(args.out_dir, "diagnosis.png")
    
    plt.figure(figsize=(12, 8))
    
    # XZ平面视图
    plt.subplot(221)
    plt.scatter(sam_aligned[:, 0], sam_aligned[:, 2], c='blue', s=1, alpha=0.5, label='Model')
    plt.scatter(part_pts[:, 0], part_pts[:, 2], c='red', s=1, alpha=0.3, label='Partial')
    plt.title('XZ平面视图 (主视图)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Y轴分布
    plt.subplot(222)
    plt.hist(sam_aligned[:, 1], bins=50, alpha=0.7, label='Model Y', color='blue')
    plt.hist(part_pts[:, 1], bins=50, alpha=0.7, label='Partial Y', color='red')
    plt.title('Y轴分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 距离热力图
    plt.subplot(223)
    sc = plt.scatter(sam_aligned[:, 0], sam_aligned[:, 2], 
                    c=dists, cmap='viridis', s=2, vmin=0, vmax=np.percentile(dists, 90))
    plt.colorbar(sc, label='距离(m)')
    plt.title('配准距离热力图')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Y轴误差图
    plt.subplot(224)
    sc = plt.scatter(sam_aligned[:, 0], sam_aligned[:, 2], 
                    c=y_axis_err, cmap='hot', s=2, vmin=0, vmax=np.percentile(y_axis_err, 90))
    plt.colorbar(sc, label='Y轴误差(m)')
    plt.title('Y轴对齐误差')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_path)
    print(f"   ✓ 诊断可视化已保存: {vis_path}")
    print("="*50)
    
    # 11. 额外诊断：保存变换矩阵
    transform_path = os.path.join(args.out_dir, "transform.json")
    transform_data = {
        "rotation_matrix": final_R.tolist(),
        "translation_vector": final_t.tolist(),
        "scale_factor": float(scale),
        "best_scheme": win_name,
        "final_fitness": float(fit_final) if 'fit_final' in locals() else 0.0
    }
    with open(transform_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    print(f"   ✓ 变换参数已保存: {transform_path}")

if __name__ == "__main__":
    main()