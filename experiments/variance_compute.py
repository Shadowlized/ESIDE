import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import DATASET_ROUTES


def compute_image_variance(image_path):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image).flatten()
    return float(np.var(image_np))


def compute_variances_in_folder(step, folder_path, output_json='variances.json'):
    variances = []
    for filename in tqdm(os.listdir(folder_path), desc=f'Step {step}'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_path = os.path.join(folder_path, filename)
            try:
                var = compute_image_variance(image_path)
                variances.append(var)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    with open(output_json, 'a') as file:
        file.write(json.dumps({"step": step, "variances": variances}) + "\n")


def compute_and_save_density(x_vals, y_vals, output_path):
    xy = np.vstack([x_vals, y_vals])
    kde = gaussian_kde(xy)
    density = kde(xy)

    result = [
        {"x": float(x), "y": float(y), "density": float(d)}
        for x, y, d in zip(x_vals, y_vals, density)
    ]

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'Density values saved to {output_path}')


def load_density_data(density_json_path):
    with open(density_json_path, 'r') as f:
        data = json.load(f)

    x_vals = np.array([item["x"] for item in data])
    y_vals = np.array([item["y"] for item in data])
    density_vals = np.array([item["density"] for item in data])

    return x_vals, y_vals, density_vals


def draw_dot_plot(name, output_json):
    # data = [{"step": 0, "variances": [2540.161249644123, 4663.682792981052, 3436.9602512812453, 5150.282214923839, 8508.10076295955]}]
    data = []
    with open(output_json, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    x_vals = []
    y_vals = []
    for item in data:
        step = item["step"]
        variances = item["variances"]
        # variances = item["variances"][:100]
        x_vals.extend([step] * len(variances))
        y_vals.extend(variances)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # dot density
    xy = np.vstack([x_vals, y_vals])
    density = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_axisbelow(True)  # z-level
    sc = ax.scatter(x_vals, y_vals, c=density, cmap='viridis', s=8, alpha=0.5, zorder=1)

    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('Variance', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, 14000)
    ax.grid(color='#dddddd', linestyle='--', linewidth=0.5)

    cbar = plt.colorbar(sc, ax=ax, label='Density')
    cbar.set_label('Density', rotation=270, labelpad=15, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.offsetText.set_fontsize(14)
    cbar.update_ticks()

    if name == 'ai':
        ax.set_title('Variance per Sample by Timestep (GLIDE)', fontsize=18)
    else:
        ax.set_title('Variance per Sample by Timestep (Natural)', fontsize=18)

    plt.tight_layout()
    plt.savefig(f'/data/ESIDE/variance_dot_plot_{name}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'/data/ESIDE/variance_dot_plot_{name}.pdf', bbox_inches='tight')
    plt.close()

    print('Figure saved')


def draw_combined_dot_plot(output_path):
    x_ai_offset, y_ai, density_ai = load_density_data('/data/ESIDE/results/ai_density.json')
    x_nature_offset, y_nature, density_nature = load_density_data('/data/ESIDE/results/nature_density.json')

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_axisbelow(True)
    ax.grid(color='#dddddd', linestyle='--', linewidth=0.5)

    sc_ai = ax.scatter(x_ai_offset, y_ai, c=density_ai, cmap='viridis', s=10, alpha=0.8, label='GLIDE')
    sc_nature = ax.scatter(x_nature_offset, y_nature, c=density_nature, cmap='coolwarm', s=10, alpha=0.8, label='Natural')

    ax.set_xlabel('Timestep', fontsize=16)
    ax.set_ylabel('Variance', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(0, 16000)
    ax.set_title('Inter-Pixel Variance per Image by Timestep', fontsize=20)
    ax.legend(fontsize=16, markerscale=3.0)

    cbar_ai = plt.colorbar(sc_ai, ax=ax, pad=0.01, fraction=0.03)
    cbar_ai.set_label('Density', rotation=270, labelpad=15, fontsize=16)
    cbar_ai.ax.tick_params(labelsize=16)
    cbar_ai.ax.yaxis.offsetText.set_fontsize(16)
    cbar_ai.update_ticks()

    cbar_nature = plt.colorbar(sc_nature, ax=ax, pad=0.04, fraction=0.03)
    cbar_nature.ax.tick_params(label1On=False, length=0)  # remove labels
    cbar_nature.set_ticks([])  # remove ticks

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'Combined plot saved to {output_path}')




if __name__ == "__main__":
    for i in range(0, 25):
        compute_variances_in_folder(i, f"{DATASET_ROUTES['SDV1.4']}/val/stepwise-noised/{i}/train-small/ai", "/your_home_dir/ESIDE/results/ai_variances.json")
        compute_variances_in_folder(i, f"{DATASET_ROUTES['SDV1.4']}/val/stepwise-noised/{i}/train-small/nature", "/your_home_dir/ESIDE/results/nature_variances.json")

    draw_combined_dot_plot(
        '/your_home_dir/ESIDE/results/ai_variances.json',
        '/your_home_dir/ESIDE/results/nature_variances.json',
        '/your_home_dir/ESIDE/variance_dot_plot_combined.pdf'
    )
