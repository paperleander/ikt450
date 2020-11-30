#!/usr/bin/env python
import subprocess
import time


scripts = ["mlp", "cnn", "lstm", "cnn_lstm", "conv_lstm", "classics"]


for script in scripts:
    start = time.time()
    subprocess.call(['python', script+'.py'])
    end = time.time()
    print(script, float(end-start))


