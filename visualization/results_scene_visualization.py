import os
import numpy as np

SCANNET_COLOR_MAP = np.array([
    [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120],
    [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40],
    [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207],
    [178, 76, 76], [247, 182, 210], [66, 188, 102], [219, 219, 141],
    [140, 57, 197], [202, 185, 52], [51, 176, 203], [200, 54, 131]
], dtype=np.uint8)

def save_ply(xyz, labels, color_map, save_path):
    black_color = np.array([0, 0, 0], dtype=np.uint8)
    colors = np.tile(black_color, (labels.shape[0], 1))
    valid = labels >= 0
    colors[valid] = color_map[np.clip(labels[valid], 0, len(color_map)-1)]

    with open(save_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property double x\nproperty double y\nproperty double z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def main():
    # scannet-tf_b-scannet
    # point/data/vox004_scale06_opacity/val
    # point/exp/scannet/scannet-tf_b-vox004_scale06_opacity/result
    base_dir = "/home/knuvi/Desktop/song"
    scene_root = os.path.join(base_dir, "point/data/vox004_scale06_opacity/val")
    result_dir = os.path.join(base_dir, "point/exp/scannet/scannet-tf_b-vox004_scale06_opacity/result")
    save_dir = os.path.join(base_dir, "gaussian-point-sampler/test/visualization")
    os.makedirs(save_dir, exist_ok=True)

    scene_ids = sorted(os.listdir(scene_root))
    for scene_id in scene_ids:
        if scene_id == 'scene0653_00':
            scene_path = os.path.join(scene_root, scene_id)
            coord_path = os.path.join(scene_path, "coord.npy")
            gt_path = os.path.join(scene_path, "segment20.npy")
            pred_path = os.path.join(result_dir, f"{scene_id}_pred.npy")
            save_path = os.path.join(save_dir, f"{scene_id}_pred.ply")

            if not (os.path.exists(coord_path) and os.path.exists(gt_path) and os.path.exists(pred_path)):
                print(f"[!] Skip {scene_id} (missing file)")
                continue

            print(f"[🔄] Processing {scene_id}...")

            coord = np.load(coord_path)
            gt = np.load(gt_path)
            pred = np.load(pred_path)
            if pred.ndim == 2:
                pred = np.argmax(pred, axis=1)
            pred = pred.astype(np.int32)

            if coord.shape[0] != pred.shape[0]:
                print(f"[❌] Shape mismatch: coord={coord.shape}, pred={pred.shape}")
                continue

            final_label = gt.copy()
            valid = gt != -1
            final_label[valid] = pred[valid]

            save_ply(coord, final_label, SCANNET_COLOR_MAP, save_path)
            print(f"[✔] Saved: {save_path}")

    print("\n[✅] 전체 시각화 완료!")


if __name__ == "__main__":
    main()
