import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

def load_image(image_path):
    if not os.path.exists(image_path):
        return np.zeros((512, 512, 3), dtype=np.uint8)  # Заглушка, если файл отсутствует
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_prompt_image(text, size=(512, 512)):
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255  # Белый фон
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_font_scale = 0.7
    min_font_scale = 0.4
    thickness = 2
    margin = 20
    max_width = size[0] - 2 * margin
    max_height = size[1] - 2 * margin
    
    lines = []
    for part in text.split("\n"):  # Разделяем строки вручную
        words = part.split()
        current_line = part.split(": ")[0] + ":"  # Добавляем заголовок строки (Orig: / Edit:)
        content = part.split(": ")[1] if ": " in part else ""
        
        for word in content.split():
            test_line = current_line + " " + word if current_line else word
            text_size = cv2.getTextSize(test_line, font, max_font_scale, thickness)[0]
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
    
    font_scale = max_font_scale
    while font_scale >= min_font_scale:
        text_height = len(lines) * (cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 10)
        if text_height <= max_height:
            break
        font_scale -= 0.05
    
    y0 = margin + (max_height - text_height) // 2 + 20
    for i, line in enumerate(lines):
        y = y0 + i * (cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 10)
        cv2.putText(img, line, (margin, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    return img

def plot_comparison(image_ids, experiment_names, base_folder="data", exp_folder="experiments/results/pie_bench", save_path="experiments/imgs_compare", save_name=None, title_fontsize=12):
    mapping_path = os.path.join(base_folder, "mapping_file.json")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    
    num_cols = len(experiment_names) + 2  # Target + Experiments + Prompts
    fig, axes = plt.subplots(len(image_ids), num_cols, figsize=(4 * num_cols, 4 * len(image_ids)))
    
    # Гарантируем, что axes всегда 2D
    if len(image_ids) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(experiment_names) + 2 == 1:
        axes = np.expand_dims(axes, axis=1)
    
    # Устанавливаем заголовки для первой строки
    axes[0, 0].set_title("Target", fontsize=title_fontsize)
    for col_idx, exp_name in enumerate(experiment_names):
        axes[0, col_idx + 1].set_title(exp_name, fontsize=title_fontsize)
    axes[0, num_cols - 1].set_title("Prompts", fontsize=title_fontsize)
    
    for row_idx, image_id in enumerate(image_ids):
        image_data = mapping.get(image_id, {})
        image_path = image_data.get("image_path", "")
        original_prompt = image_data.get("original_prompt", "")
        editing_prompt = image_data.get("editing_prompt", "")
        
        # Target изображение
        target_path = os.path.join(base_folder, "annotation_images", image_path)
        target_img = load_image(target_path)
        axes[row_idx, 0].imshow(target_img)
        axes[row_idx, 0].axis("off")
        
        # Колонки экспериментов
        for col_idx, exp_name in enumerate(experiment_names):
            exp_image_path = os.path.join(exp_folder, exp_name, "annotation_images", image_path)
            exp_img = load_image(exp_image_path)
            axes[row_idx, col_idx + 1].imshow(exp_img)
            axes[row_idx, col_idx + 1].axis("off")
        
        # Колонка с промптами в виде изображения
        prompt_text = f"Orig: {original_prompt}\nEdit: {editing_prompt}\nImg_id: {image_id}"
        prompt_img = create_prompt_image(prompt_text)
        axes[row_idx, num_cols - 1].imshow(prompt_img)
        axes[row_idx, num_cols - 1].axis("off")
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(f"{save_path}/{save_name}", dpi=300, bbox_inches='tight')
    
    plt.show()

def sample_images_by_editing_type(mapping_file, num_samples_per_type, editing_types=None):
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    
    if editing_types is None:
        editing_types = list(map(str, range(10)))  # По умолчанию выбираем типы от 0 до 9
    
    images_by_type = {etype: [] for etype in editing_types}
    for image_id, data in mapping.items():
        editing_type = str(data.get("editing_type_id", ""))
        if editing_type in images_by_type:
            images_by_type[editing_type].append(image_id)
    
    sampled_images = []
    for etype, images in images_by_type.items():
        sampled_images.extend(random.sample(images, min(num_samples_per_type, len(images))))
    
    return sampled_images



experiment_names = [
    "origin_params_InfEdit"
]

title_image_ids = ["000000000000", "211000000003", "511000000001"]
plot_comparison(title_image_ids, experiment_names, save_name="title_images.png")

random_image_ids = sample_images_by_editing_type("data/mapping_file.json", num_samples_per_type=1, editing_types=None)
plot_comparison(random_image_ids, experiment_names, save_name="random_images.png")

