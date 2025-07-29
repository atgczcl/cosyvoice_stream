# 一个模拟运行的python文件，用于测试cosyvoice_stream的运行环境

import time

# 10秒钟打印一次"Hello, world!", 10秒钟后退出
a = 0
while True:
    time.sleep(2)
    print("Hello, world!", a)
    a += 1
    if a == 10:
        break