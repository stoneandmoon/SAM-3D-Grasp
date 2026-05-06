import numpy as np
import trimesh

ply_path = "output_canonical/aligned_source_to_partial.ply" 
out_npy = "my_blue_bottle.npy"  # 👈 直接保存在当前目录下！

print(f"正在加载点云: {ply_path}")
try:
    pcd = trimesh.load(ply_path)
    points = np.array(pcd.vertices).astype(np.float32)
    
    np.save(out_npy, points)
    print(f"✅ 转换成功！保存至 {out_npy}")
    print(f"📊 点云形状: {points.shape}")
except Exception as e:
    print(f"❌ 转换失败: {e}")