import os
import cv2
import joblib
import shutil
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
from PIL import Image

def spline_interpolate(points=None):
    if isinstance(points, list):
        points = np.stack(points, axis=0)
    if points.ndim > 2:
        raise ValueError('points must be a 2d array or 1d array list')
    x, y = points[:, 0], points[:, 1]
    num_points = points.shape[0]
    t = np.linspace(0, 1, num_points)
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)
    t_dense = np.linspace(0, 1, 500)
    x_dense = spline_x(t_dense)
    y_dense = spline_y(t_dense)
    return x, y, x_dense, y_dense


def add_random_noise_to_contours(contours, max_displacement):
    noisy_contours = []
    for contour in contours:
        noisy_contour = []
        for point in contour:
            x, y = point[0]
            dx = np.random.randint(-max_displacement, max_displacement)
            dy = np.random.randint(-max_displacement, max_displacement)
            noisy_contour.append([[x + dx, y + dy]])
        noisy_contours.append(np.array(noisy_contour, dtype=np.int32))
    return noisy_contours


def distort_image(image, output_path, max_displacement=2):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    noisy_contours = add_random_noise_to_contours(contours, max_displacement)

    height, width = image.shape
    result_image = np.ones((height, width), dtype=np.uint8) * 255

    cv2.drawContours(result_image, noisy_contours, -1, (0), cv2.FILLED)

    cv2.imwrite(output_path, result_image)


def copy_images():
    des_dir = 'data/train_sketches'
    os.makedirs(des_dir, exist_ok=True)
    src_dir = 'data/train_sketch'
    seq_dirs = os.listdir(src_dir)
    for seq_idx, seq_dir in tqdm(enumerate(seq_dirs)):
        frame_dirs = sorted(os.listdir(os.path.join(src_dir, seq_dir)))
        for frame_idx, frame_dir in enumerate(frame_dirs):
            src_filename = f'{src_dir}/{seq_dir}/{frame_dir}/1.png'
            des_filename = f'{des_dir}/{int(seq_dir):03d}_{int(frame_dir):04d}.png'
            shutil.copy(src_filename, des_filename)
    return


# input: [10,22,3]
def draw_sketches(seq, save_dir, i, base_sketch=None, color="black"):
    matplotlib.use('Agg')
    connections = [[0, 2, 5, 8, 11],
                   [0, 1, 4, 7, 10],
                   [0, 3, 6, 9, 12, 15],
                   [9,14,17,19,21],
                    [9,13,16,18,20]]
    # connections = [[5, 4, 22, 0, 1],
    #                [22, 8, 9, 10, 11],
    #                [16, 14, 13, 12, 9, 17, 18, 19, 21]]

    joints_bezier = seq
    num_frames = joints_bezier.shape[0]
    joints_bezier = joints_bezier.reshape(num_frames, -1, 3)
    joints_bezier = joints_bezier[..., :2]
    joints_bezier = joints_bezier + 500
    joints_bezier = joints_bezier.clip(0, 999) / 4

    hips = (joints_bezier[:, 0:1, :] + joints_bezier[:, 4:5, :]) / 2
    waists = (hips + 2 * joints_bezier[:, 8, :]) / 3
    joints_bezier = np.concatenate((joints_bezier, waists), axis=1)

    for k in tqdm(range(num_frames)):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
        for l, connection in enumerate(connections):
            x, y, x_dense, y_dense = spline_interpolate(joints_bezier[k, connection])
            plt.plot(x_dense, y_dense, linewidth=4.0, color=color)
            # plt.scatter([x_dense[0], x_dense[-1]], [y_dense[0], y_dense[-1]], c='red', s=40, zorder=10)

        plt.tight_layout()
        plt.axis('off')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        save_path=f'{save_dir}/{i:04d}_{k:04d}.png'
        fig.savefig(save_path, dpi=100, transparent=True)
        plt.close(fig)
        if base_sketch is not None:
            overlay = Image.open(save_path).convert("RGBA").resize((128, 128))

            sketch_img = base_sketch[k].detach().cpu().permute(1, 2, 0).numpy()
            sketch_img = (sketch_img * 255).astype(np.uint8) if sketch_img.max() <= 1.0 else sketch_img
            sketch_img = Image.fromarray(sketch_img).convert("RGBA")

            # Alpha 合成叠加关节点图
            composed = Image.alpha_composite(sketch_img, overlay)
            composed.save(save_path)

def draw_joints(seq, save_dir, i, base_sketch=None, color="black"):
    import matplotlib
    matplotlib.use('Agg')

    # 定义骨骼连接结构
    connections = [[0, 2, 5, 8, 11],
                   [0, 1, 4, 7, 10],
                   [0, 3, 6, 9, 12, 15],
                   [9,14,17,19,21],
                    [9,13,16,18,20]]

    # 预处理关节点数据
    seq = seq.detach().cpu().numpy() if torch.is_tensor(seq) else seq
    num_frames = seq.shape[0]
    joints = seq[..., :2]  # 只取 x, y
    joints = joints + 500
    joints = joints.clip(0, 999) / 4

    # 添加腰部关节（作为第 23 个点）
    hips = (joints[:, 0:1, :] + joints[:, 4:5, :]) / 2
    waists = (hips + 2 * joints[:, 8:9, :]) / 3
    joints = np.concatenate((joints, waists), axis=1)  # shape: [B, J+1, 2]

    # 绘图
    for k in tqdm(range(num_frames), desc="Drawing sketches"):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)

        # 画骨架
        for connection in connections:
            pts = joints[k, connection]  # shape: [N, 2]
            ax.plot(pts[:, 0], pts[:, 1], linewidth=2.0, color=color)

        # 画关节点
        ax.scatter(joints[k, :, 0], joints[k, :, 1], s=10, c='red')

        ax.axis('off')

        os.makedirs(save_dir, exist_ok=True)
        save_path=f'{save_dir}/{i:03d}_{k:04d}.png'
        fig.savefig(save_path, dpi=100, transparent=True)
        plt.close(fig)
        if base_sketch is not None:
            overlay = Image.open(save_path).convert("RGBA").resize((128, 128))

            sketch_img = base_sketch[k].detach().cpu().permute(1, 2, 0).numpy()
            sketch_img = (sketch_img * 255).astype(np.uint8) if sketch_img.max() <= 1.0 else sketch_img
            sketch_img = Image.fromarray(sketch_img).convert("RGBA")

            # Alpha 合成叠加关节点图
            composed = Image.alpha_composite(sketch_img, overlay)
            composed.save(save_path)
        


if __name__ == '__main__':
    connections = [[0, 2, 5, 8, 11],
                   [0, 1, 4, 7, 10],
                   [0, 3, 6, 9, 12, 15],
                   [9,14,17,19,21],
                    [9,13,16,18,20]]

    save_dir = f'dataset/sketches'
    os.makedirs(save_dir, exist_ok=True)
    total_frames = 0
    linewidth = 6.0

    matplotlib.use('Agg')
    data_dict = joblib.load(f'dataset/data_dict_xyz.pkl') #[150,692,22,3] [clip,22,3,length]
    name_list=joblib.load(f'dataset/name_list.pkl')
    num_seqs = len(data_dict)
    num_frames_list = []
    for name in tqdm(name_list, desc="Processing motions:"):
        seq= data_dict[name]['motion']  # [22, 3, length]
        seq = seq.transpose(2, 0, 1)
        num_frames = seq.shape[0]
        num_frames_list.append(num_frames)
        total_frames += num_frames

        # joints_bezier = train_seqs[i].numpy()
        joints_bezier = seq
        num_frames = joints_bezier.shape[0]
        joints_bezier = joints_bezier.reshape(num_frames, -1, 3)
        joints_bezier = joints_bezier[..., :2]
        joints_bezier = joints_bezier + 500
        joints_bezier = joints_bezier.clip(0, 999) / 4

        hips = (joints_bezier[:, 0:1, :] + joints_bezier[:, 4:5, :]) / 2
        waists = (hips + 2 * joints_bezier[:, 8, :]) / 3
        joints_bezier = np.concatenate((joints_bezier, waists), axis=1)

        # hips = (joints_bezier[:, 0, :] + joints_bezier[:, 4, :]) / 2
        # waists = (hips + 2 * joints_bezier[:, 8, :]) / 3
        # left_ankles = joints_bezier[:, 1, :]
        # right_ankles = joints_bezier[:, 5, :]
        #
        # head_tops = joints_bezier[:, 11, :]
        # necks = joints_bezier[:, 9, :]
        #
        # left_hands = joints_bezier[:, 19, :]
        # right_hands = joints_bezier[:, 14, :]
        # first line: left_ankles -> waists -> right_ankles
        # second line: waists -> necks -> head_tops
        # third line: left_hands -> necks -> right_hands

        for j in tqdm(range(num_frames)):
            # write_full_dir = os.path.join(save_dir, str(i), str(j)).replace('\\', '/')
            # write_full_path = '{0}/{1}.png'
            # os.makedirs(write_full_dir, exist_ok=True)
            # # first we get the number of files
            # num_files = len(os.listdir(write_full_dir))

            # first line: left_ankles -> waists -> right_ankles
            # second line: waists -> necks -> head_tops
            # third line: left_hands -> necks -> right_hands
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            for k, connection in enumerate(connections):
                x, y, x_dense, y_dense = spline_interpolate(joints_bezier[j, connection])
                plt.plot(x_dense, y_dense, linewidth=linewidth, color='black')
            # x, y, x_dense, y_dense = spline_interpolate([left_ankles[j], waists[j], right_ankles[j]])
            # ax.plot(x_dense, y_dense, linewidth=linewidth, color='black')
            # x, y, x_dense, y_dense = spline_interpolate([waists[j], necks[j], head_tops[j]])
            # ax.plot(x_dense, y_dense, linewidth=linewidth, color='black')
            # x, y, x_dense, y_dense = spline_interpolate([left_hands[j], necks[j], right_hands[j]])
            # ax.plot(x_dense, y_dense, linewidth=linewidth, color='black')
            plt.tight_layout()
            plt.axis('off')

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            # smoothed_sketch = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            # width, height = canvas.get_width_height()
            # smoothed_sketch = smoothed_sketch.reshape(height, width, 3)

            # num_files += 1
            # fig.savefig(write_full_path.format(write_full_dir, num_files), dpi=100)
            # fig.savefig(f'{save_dir}/{i:03d}_{j:04d}.png', dpi=100)
            
            fig.savefig(f'{save_dir}/{name}_{j:04d}.png', dpi=100)

            plt.close(fig)

            # # save the smoothed sketch and its 3 different versions
            # binary_smoothed_sketch = cv2.cvtColor(smoothed_sketch, cv2.COLOR_RGB2GRAY)
            # for k in range(3):
            #     num_files += 1
            #     distort_image(binary_smoothed_sketch.copy(), write_full_path.format(write_full_dir, num_files), 1)
            #     num_files += 1
            #     distort_image(binary_smoothed_sketch.copy(), write_full_path.format(write_full_dir, num_files), 2)
            #     num_files += 1
            #     distort_image(binary_smoothed_sketch.copy(), write_full_path.format(write_full_dir, num_files), 3)
