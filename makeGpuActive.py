import torch
import time
import random
from tqdm import tqdm
import math

# Настройка прогресс-бара
progress_bar = tqdm(
    desc="GPU Working", 
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    dynamic_ncols=True
)

# Бесконечный цикл с матричными операциями
iteration = 0
while True:
    numb = 1024 * 4
    x = torch.randn(numb, numb, device='cuda')
    y = torch.randn(numb, numb, device='cuda')
    
    start_time = time.time()
    z = x @ y  # Матричное умножение
    torch.cuda.synchronize()  # Гарантированная нагрузка
    elapsed = time.time() - start_time
    
    # Обновляем прогресс-бар
    iteration += 1
    progress_bar.set_postfix({
        'iter': iteration,
        'time': f"{elapsed:.3f}s",
        'GFLOPS': f"{(2 * numb**3 / (elapsed * 1e9)):.1f}"  # Теоретическая производительность
    })
    progress_bar.update(1)
    
    time.sleep(0.01)  # Не перегревать GPU