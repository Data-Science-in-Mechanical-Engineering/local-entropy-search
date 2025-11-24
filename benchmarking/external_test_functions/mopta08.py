import os
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine
import time

import numpy as np

# this function needs to be minimized! (see https://web.archive.org/web/20200502001659/https://anjos.mgi.polymtl.ca/MOPTA2008Benchmarkdata/mopta08_jones.pdf slide 6) and the domain is within [0 1] 

def objective(x):
	a = time.time()
	mopta_exectutable = "mopta08_elf64.bin"
	mopta_full_path = os.path.join(
		Path(__file__).parent, "mopta08", mopta_exectutable
	)

	directory_file_descriptor = tempfile.TemporaryDirectory()
	directory_name = directory_file_descriptor.name

	with open(os.path.join(directory_name, "input.txt"), "w+") as tmp_file:
		for _x in x:
			tmp_file.write(f"{_x}\n")
	popen = subprocess.Popen(
		mopta_full_path,
		stdout=subprocess.PIPE,
		cwd=directory_name,
	)
	popen.wait()
	output = (
		open(os.path.join(directory_name, "output.txt"), "r")
		.read()
		.split("\n")
	)
	output = [x.strip() for x in output]
	output = np.array([float(x) for x in output if len(x) > 0])
	value = output[0]
	constraints = output[1:]
	print(value, constraints)

	b = time.time()
	elapsed_time = b-a
	print(f"Time for mopta call: {elapsed_time}")
	sys.stdout.flush()
	return (value + 10 * np.sum(np.clip(constraints, a_min=0, a_max=None)))

if __name__ == '__main__':

	x = np.random.rand(124)
	value = objective(x)
	print(f"parameters: {x}")
	print(f"value (to be minimized): {value}")
	