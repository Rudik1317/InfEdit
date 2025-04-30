import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

import math
from PIL import Image


def create_image_grid(folder_path, output_path):
    # Получаем список изображений в папке
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not images:
        print("Нет изображений в указанной папке.")
        return
    
    # Открываем все изображения и получаем их размеры
    opened_images = [Image.open(img) for img in images]
    widths, heights = zip(*(img.size for img in opened_images))
    
    # Определяем количество колонок и строк
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    
    # Определяем размеры итогового изображения
    max_width = max(widths)
    max_height = max(heights)
    total_width = cols * max_width
    total_height = rows * max_height
    
    # Создаём пустое изображение
    output_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Располагаем изображения в сетке
    for idx, img in enumerate(opened_images):
        x_offset = (idx % cols) * max_width
        y_offset = (idx // cols) * max_height
        output_image.paste(img, (x_offset, y_offset))
    
    # Сохраняем итоговое изображение
    output_image.save(output_path)
    print(f"Готово! Изображение сохранено в {output_path}")

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
    """
    строит график где колонки это названия експерименитов (первая колонка это базовое изображение, последняя промт)
    строка принадлежит одной картинке
    На вход 
        адишники картинок которые нужно
        базовая папка откуда забираем сорс и промты
        папка где лежат папки експериментов с итоговыми картинками
        путь куда сохраняем картинку итог
        название картинки итог
        размер текста
    """
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
    axes[0, 0].set_title("Source", fontsize=title_fontsize)
    for col_idx, exp_name in enumerate(experiment_names):
        axes[0, col_idx + 1].set_title(exp_name, fontsize=title_fontsize)
    axes[0, num_cols - 1].set_title("Prompts", fontsize=title_fontsize)
    
    for row_idx, image_id in enumerate(image_ids):
        image_data = mapping.get(image_id, {})
        image_path = image_data.get("image_path", "")
        original_prompt = image_data.get("original_prompt", "")
        editing_prompt = image_data.get("editing_prompt", "")
        blended_word = image_data.get("blended_word", "")
        
        # Target изображение
        target_path = os.path.join(base_folder, "annotation_images", image_path)
        target_img = load_image(target_path)
        axes[row_idx, 0].imshow(target_img)
        axes[row_idx, 0].axis("off")
        
        # Колонки экспериментов
        for col_idx, exp_name in enumerate(experiment_names):
            exp_image_path = os.path.join(exp_folder, exp_name, "annotation_images", image_path)
            if not os.path.exists(exp_image_path):
                print(f"Image not found: {exp_image_path}")
                return
            exp_img = load_image(exp_image_path)
            axes[row_idx, col_idx + 1].imshow(exp_img)
            axes[row_idx, col_idx + 1].axis("off")
        
        # Колонка с промптами в виде изображения
        prompt_text = f"Orig: {original_prompt}\nEdit: {editing_prompt}\nMutual Blend: {blended_word} \nImg_id: {image_id}"
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



other_title_image_ids = [
    "other_title_image_ids",
    "000000000007", #dog
    "612000000003", #girl umbrella
    "121000000001", #cattiger
    "511000000002", #bear
    ]
figure_9_images = [
    "figure_9_images",
    "111000000001",
    "112000000009",
    "123000000005",
    "124000000008"
]

figure_10_images = [
    "figure_10_images",
    "221000000002",
    "222000000001",
    "213000000005",
    "214000000000",
]

figure_11_images = [
    "figure_11_images",
    "311000000002",
    "322000000000",
    "313000000009",
    "324000000005",
]

figure_12_images = [
    "figure_12_images",
    "411000000004",
    "422000000000",
    "414000000003",
    "423000000000",
]

figure_13_images = [
    "figure_13_images",
    "511000000004",
    "512000000001",
    "523000000003",
    "524000000002"
]

figure_14_images = [
    "figure_14_images",
    "621000000000",  # a small white blue lamb standing in the grass
    "622000000001",  # a woman in a white red dress sitting on a chair with flowers
    "623000000002",  # a red green lipstick is being splashed with red powder
    "614000000001" 
]

figure_15_images = [
    "figure_15_images",
    "721000000000",  # a photo of a bronze horse in the field
    "712000000001",  # a drawing of a young robot with blue eyes
    "723000000003",  # a chocolate icecream cake with candies on top
    "714000000003",  # the golden crescent moon and stars are seen in the night sky
]

figure_16_images = [
    "figure_16_images",
    "811000000009",
    "812000000000",
    "823000000006",
    "823000000007"
]

figure_17_images = [
    "figure_17_images",
    "911000000001",
    "922000000000",
    "923000000002",
    "924000000001"
]

my_sample_images = [
    "121000000001",
    "221000000002",
    "311000000002",
    "411000000004",
    "511000000002",
    "621000000002",
    "712000000002",
    "812000000000",
    "922000000000",
]
my_sample_images2 =[
        "213000000005",
        "000000000009",
        "712000000001",
        "621000000001",
        "422000000000",
        "523000000001",
]

##### рандомная сверка изображений
def random_images_compare():
    random_image_ids = sample_images_by_editing_type("data/mapping_file.json", num_samples_per_type=4, editing_types=None)
    print(random_image_ids)
    plot_comparison(random_image_ids, experiment_names, save_name=f"random.png")

##############################
experiment_names = [
    "origin_params_InfEdit",
    "set_attn_base_store_like_in_app",
    "set_title_MSAC",
    "set_title_MSAC_mutula_is_mutual",
]
IMAGES = [
    #random_image_ids,
    other_title_image_ids,
    figure_9_images,
    figure_10_images,
    figure_11_images,
    figure_12_images,
    figure_13_images,
    figure_14_images,
    figure_15_images,
    figure_16_images,
    figure_17_images,
]

def batch_images_compare(IMAGES, experiment_names):
    for i in IMAGES:
        plot_comparison(
            i[1:], 
            experiment_names, 
            save_name=f"{i[0]}.png",
            exp_folder="experiments/imgs_results/pie_bench",
            save_path="experiments/imgs_compare/see_diff_code_and_title"
        )

def grid_images_compare():
    grid_parameters = {
        "guidance_s": [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 3],
        "guidance_t": [1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 3, 5],
        "seed": [0, 17, 2342, 34534, 342534, 34985383, 89459499],
        "cross_replace_steps": [0, 0.1, 0.3, 0.5, 0.7, 0.9 ,1],
        "self_replace_steps": [0, 0.1, 0.3, 0.5, 0.7, 0.9 ,1],
        "thresh_e": [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.9, 1],
        "thresh_m": [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.9, 1],
        "strength": [0.75, 0.85, 0.88, 0.9, 0.93, 0.96, 0.98, 1, 0],
        "eta": [0.3, 0.5, 0.75, 1, 1.2, 1.5],
    }

    for param in grid_parameters:
        plot_comparison(
            my_sample_images, 
            experiment_names=[f"{i}_{param}" for i in grid_parameters[param]], 
            save_name=f"{param}.png",
            exp_folder=f"experiments/imgs_results/{param}",
            save_path="experiments/imgs_compare/try_2702"
        )

def tmp():
    image_ids = [
        ["000000000007", "000000000009", "111000000001", "112000000009", "121000000001"],
        ["123000000005", "124000000008", "213000000005", "214000000000", "221000000002"],
        ["222000000001", "311000000002", "313000000009", "322000000000", "324000000005"],
        ["411000000004", "414000000003", "422000000000", "423000000000", "511000000002"],
        ["511000000004", "512000000001", "523000000001", "523000000003", "524000000002"],
        ["612000000003", "614000000001", "621000000000", "621000000001", "621000000002"],
        ["622000000001", "623000000002", "712000000001", "712000000002", "721000000000"],
        ["723000000003", "714000000003", "811000000009", "812000000000", "823000000006"],
        ["823000000007", "911000000001", "922000000000", "923000000002", "924000000001"]
    ]
    for idx, i in enumerate(image_ids):
        plot_comparison(
            i, 
            experiment_names=["origin_params_InfEdit", "cross_src_scac"], 
            save_name=f"{idx}.png",
            exp_folder="experiments/imgs_results",
            save_path="experiments/imgs_compare/cross_src_scac"
        )
            
    # Пример использования
    create_image_grid("experiments/imgs_compare/cross_src_scac", "experiments/imgs_compare/cross_src_scac_sum.jpg")