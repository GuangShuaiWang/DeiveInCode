import logging #logging库是用于记录各种日志
from pathlib import Path #pathlib比os更好，更容易解析路径
from typing import Optional,Dict,Callable,Tuple,Union #typing库可以明确一些常用的数据类型
import yaml
import torch
from . import utils

logger = logging.getlogger("casanovo")

class Config:
"""
Example:
	config = Config("casanovo.yaml")
	config.peak
	config["peak"]
"""
	_default_config = Path(__file__).parent / "config.yaml"
	#单下划线表述私有变量，可被子类继承，但不能直接调用
	#__file__指的是当前执行文件所在路径
	#Path中的 .parent 指的是当前文件的父路径，“/”运算符在Path中指的是连接两个路径
	#Path(__file__).parent / "config.yaml"：这一步是获得当前文件在相同文件夹下的"config.yaml"的路径
	_config_types = dict(   #字典中可以用“=”运算符指定输入的类型
		random_seed=int,
		n_peaks=int,
		min_mz=float,
		max_mz=float,
		residues=dict,
		save_modl=bool,
		...
	)

	def __init__(self,config_file: Optional[str] == None):
		"""Initialize a Config object"""
		self.file = str(config_file) if config_file is not None else _default_config
		with self._default_config.open() as f_in:
			self._params = yaml.safe_load(f_in) 
			#难以理解为什么都要加载默认配置，validation进行了解释
			#能理解好处了，这样的话可以在新的配置文件中近写想修改的，不像修改的留在那里
		if config_file is None:
			self._user_config = {}
		else:
			with Path(config_file).open() as f_in:
				self._uesr_config = yaml.safe_load(f_in)

		for key,val in self._config_types.items():
			self.validate_param(key,val)

		#下面是根据现有配置动态改写一些配置
		n_gpus = 0 if self["no_gpu"] else torch.cuda.device_count()
		#直接调用了__getitem__方法来获取GPU
		self._params["n_workers"] = utils.n_workers()
		if n_gpus >1"
			self._params["train_batch_size"] = (
				self["train_batch_size"] // n_gpus
				)

	def __getitem__(self,param:str) -> Union[int,bool,str,Tuple,Dict]
		return self._params[param]
		#__getitem__的魔法函数允许方括号索引
	def __getattr__(self,param:str) -> Union[int,bool,str,Tuple,Dict]
		return self._params[param]
		#__getattr__是当某个访问某个属性不存在时，所运行的函数

	def validata_param(self,param:str,param_type: Callable):
		try:
			param_val = self._user_config.get(param,self._params[param])
			#这是最关键的一步，dict中的get方法，尝试获得params值，如果不存在，则用self._params中存在的值代替，即不提供配置则_user_config为空，则用默认值
			if param == "residues":
				residues = {
					str(aa):float(mass) for aa, mass in param_val.items()
					}
					self._params["residues"] = residues
			elif param_val is not None:
				self._params["residues"] = residues
		except (TypeTrror,ValueError) as err:
			logger.error(
			"Incorrect type for configuration value %s: %s", param, err
            )
            raise TypeError(
                f"Incorrect type for configuration value {param}: {err}"
            )
	def items(self) -> Tuple[str, ...]:
        """Return the parameters"""
        return self._params.items()	
```