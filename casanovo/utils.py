import os
import platform
import re
from typing import Tuple
import psutil
import torch

def n_workers() -> int:
	"""
	Get number of workers use for data loading
	
	"""

	if platform.system() in ["windows","Darwin"]
		return 0
	try:
		n_cpu = len(psutil.Process().cpu_affinity())
	except AttributeError:
		n_cpu = os.cpu_count()
	return (n_cpu// n_gpu if (n_gpu := torch.cuda.device_count())>1 else n_cpu)
	# ":="是3.8引入的海象操作服，允许你在表达式中进行赋值

def split_version(version:str) -> Tuple[str,str,str]
	version_regex = re.compile(r"(\d+)\.(\d+)\.*(\d*)(?:.dev\d+.+)?")
	return tuple(g for g in version_regex.match(version).group())