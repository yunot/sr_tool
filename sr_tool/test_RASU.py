# @Author: yunotao :)
# @Date: 9/6/2023 :)
import subprocess
import time

command = 'FidelityFX_CLI.exe -Mode RASU -Scale 2x 2x resources/images/128x128_1.jpg results/image_sr/image_sgsr/128x128_1_SGSR.jpg'
itime = 0

for _ in range(10):
    start_time = time.perf_counter()
    subprocess.run(command, shell=True, encoding='utf-8')
    duration = time.perf_counter() - start_time
    print(f"RASU Duration: {duration}")

    itime += duration

print(f"Avg RASU Duration: {itime / 100}")
# Avg RASU Duration 128x128 to 256x256: 0.17353455300000012
# 582 x 297 scaled to 1164 x 594:  0.18019898499999992
