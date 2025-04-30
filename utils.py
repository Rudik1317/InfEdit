import json
import os

def is_google_colab():
    try:
        import google.colab
        return True
    except:
        return False

def func_to_check_reload_import_in_ipynb(number):
    return number-number

def load_benchmark_pie_bench(
    image_ids=None,
    path_to_mapping_json='/home/mdnikolaev/darudenkov/InfEdit/data/mapping_file.json', 
    path_to_images='/home/mdnikolaev/darudenkov/InfEdit/data/annotation_images', 
):
    """
    Load the PIE-Bench benchmark dataset from JSON mapping file and image directory.
    
    Args:
        image_ids: Optional list of image IDs to filter by. If None, all images are included.
        path_to_mapping_json: Path to the JSON file containing the mapping information
        path_to_images: Path to the directory containing the images
        
    Out:
        {
            000000000007: {
                image_id: 000000000007
                image_path:"/0_random_140/000000000007.jpg"
                original_prompt:"a german shepherd dog stands on the grass with mouth [closed]"
                editing_prompt:"a german shepherd dog stands on the grass with mouth [opened]"
                editing_instruction:"Change the german shepherd dog's mouth from closed to opened"
                editing_type_id:"0"
                blended_word:"mouth mouth"
                }
        }
    """
    # Load the mapping JSON file
    with open(path_to_mapping_json, 'r') as f:
        mapping_data = json.load(f)
    
    benchmark = {}
    
    for image_id, data in mapping_data.items():
        if image_ids is not None and image_id not in image_ids:
            continue

        # Update some info
        full_img_path = os.path.join(path_to_images, data['image_path'])
        data['image_path'] = full_img_path
        data['image_id'] = image_id
        
        benchmark[image_id] = data

    return benchmark

def save_inference_image(image,
                         image_name,
                         image_ext,
                         save_path,
                         save_doubles_images_w_postfix=False,
                         verbose=False,
                        ):
    """
    image: PIL.Image объект — картинка
    image_name: название файла без расширения
    image_ext: расширение файла (например, '.png' или '.jpg')
    save_path: путь до папки, где сохранить
    save_doubles_images_w_postfix: если картинка уже существует в папке, добавить постфикс
    verbose: печатать сообщение о сохранении
    """
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{image_name}{image_ext}"
    save_full_path = os.path.join(save_path, file_name)

    if save_doubles_images_w_postfix:
        i = 1
        while os.path.exists(save_full_path):
            file_name = f"{image_name}_({i}){image_ext}"
            save_full_path = os.path.join(save_path, file_name)
            i += 1

    image.save(save_full_path)
    if verbose:
        print(f"✅ Изображение сохранено в: {save_full_path}")
    return


from PIL import Image

def concat_images_horizontally(list_images):
    """
    Склеивает список квадратных PIL изображений по горизонтали в одно.

    Args:
        list_images (list): список изображений (PIL.Image)

    Returns:
        PIL.Image: склеенное изображение
    """
    if not list_images:
        raise ValueError("Список изображений пустой!")

    # Предполагаем, что все изображения одного размера
    width, height = list_images[0][0].size

    # Создаем новое изображение подходящего размера
    total_width = width * len(list_images)
    new_image = Image.new('RGB', (total_width, height))

    # Вставляем изображения друг за другом
    x_offset = 0
    for img in list_images:
        new_image.paste(img[0], (x_offset, 0))
        x_offset += width

    return new_image

