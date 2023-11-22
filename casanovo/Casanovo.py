import datetime
import functools
import logging
import os
import re
import shutil #shell工具，文件操作接口
import sys
import warnings #用于警告控制
from typing import Optional,Tuple

warnings.fileterwarnings("ignore", category=DeprecationWarning)#忽略弃用警告

import appdirs #在不同的平台上找到合适的存储位置，这里是用来找cache文件夹
import click #非常好的包，用于写命令行
import github #与Github进行交互
import requests
import torch
import tqdm
import yaml
import pytorch_lightning.lite import LightningLite #轻量型的组件

from . import __version__
from . imprt utils
from .data import ms_io
from .denovo import model_runner
from .config import Config

logger = logging.getLogger("casanovo")

# "@"运算符定义了一个装饰器，可以在不改变代码结构增加功能，装饰器可以接受函数或者参数
@click.command() #定义为命令行命令
@click.option("--mode",required=True,default="denovo",
			  type = click.Choice(["denovo","train","eval"]),
			  help="")  #type表示必须从三项中选择
@click.option("--model", help = "",
			 type = click.Path(exists=True, dir_okay=False)) 
			 #文件必须存在，且不能为目录
...
def main(
	mode: str,
	model: Optional[str],
	peak_path: str,
	peak_path_val: Optional[str],
    config: Optional[str],
    output: Optional[str],
):
	if output is None:
		output = os.path.join(
		os.getcwd(),
		f"casanovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
		# 有问题，前后不协调，为啥不用pathlib呢？
	else:
		basename, ext = os.path.splitext(os.path.abspath(output))
        output = basename if ext.lower() in (".log", ".mztab") else output

	# Configure logging
	logging.captureWarnings(True) #将python的警告放到logging中
	root = logging.getLogger() #根记录器，全局的配置
	root.setLevel(logging.DEBUG) #记录DEBUG以上的所有日志信息
	log_formatter = logging.Formatter(
		"{asctime}{levelname}[{name}/{processName}] {module}.{funcName} : "
		"{message}", style = "{"
	) #定义log的输出格式为：时间戳，日志级别，记录器名称，进程名称，模块名和函数名，实际的信息
	# 下面定义的是控制台日志处理器，会把日志输出到 标准错误流(stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    # 下面是文件日志处理器，保存在output.log的路径里
    file_handler = logging.FileHandler(f"{output}.log")
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    # 对一些特殊的包信息进行处理
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

	# Log the active configuration.
    logger.info("Casanovo version %s", str(__version__))
    logger.debug("mode = %s", mode)
    logger.debug("model = %s", model)
    logger.debug("peak_path = %s", peak_path)
    logger.debug("peak_path_val = %s", peak_path_val)
    logger.debug("config = %s", config.file)
    logger.debug("output = %s", output)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))
    
	config = Config(config)
	LightningLite.seed_everything(seed=config["random_seed"], workers=True) 
	#这里设置的是全局种子

	if model is None and mode != "train":
        try:
            model = _get_model_weights()
        except github.RateLimitExceededException:
            logger.error(
                "GitHub API rate limit exceeded while trying to download the "
                "model weights. Please download compatible model weights "
                "manually from the official Casanovo code website "
                "(https://github.com/Noble-Lab/casanovo) and specify these "
                "explicitly using the `--model` parameter when running "
                "Casanovo."
            )
            raise PermissionError(
                "GitHub API rate limit exceeded while trying to download the "
                "model weights"
            ) from None

def _get_model_weights() -> str:
    """
    Use cached model weights or download them from GitHub.

    If no weights file (extension: .ckpt) is available in the cache directory,
    it will be downloaded from a release asset on GitHub.
    Model weights are retrieved by matching release version. If no model weights
    for an identical release (major, minor, patch), alternative releases with
    matching (i) major and minor, or (ii) major versions will be used.
    If no matching release can be found, no model weights will be downloaded.

    Note that the GitHub API is limited to 60 requests from the same IP per
    hour.

    Returns
    -------
    str
        The name of the model weights file.
    """
    cache_dir = appdirs.user_cache_dir("casanovo", False, opinion=False)
    os.makedirs(cache_dir, exist_ok=True)
    version = utils.split_version(__version__)
    version_match: Tuple[Optional[str], Optional[str], int] = None, None, 0
    # Try to find suitable model weights in the local cache.
    for filename in os.listdir(cache_dir):
        root, ext = os.path.splitext(filename)
        if ext == ".ckpt":
            file_version = tuple(
                g for g in re.match(r".*_v(\d+)_(\d+)_(\d+)", root).groups()
            )
            match = (
                sum(m)
                if (m := [i == j for i, j in zip(version, file_version)])[0]
                else 0
            )
            if match > version_match[2]:
                version_match = os.path.join(cache_dir, filename), None, match
    # Provide the cached model weights if found.
    if version_match[2] > 0:
        logger.info(
            "Model weights file %s retrieved from local cache",
            version_match[0],
        )
        return version_match[0]
    # Otherwise try to find compatible model weights on GitHub.
    else:
        repo = github.Github().get_repo("Noble-Lab/casanovo")
        # Find the best matching release with model weights provided as asset.
        for release in repo.get_releases():
            rel_version = tuple(
                g
                for g in re.match(
                    r"v(\d+)\.(\d+)\.(\d+)", release.tag_name
                ).groups()
            )
            match = (
                sum(m)
                if (m := [i == j for i, j in zip(version, rel_version)])[0]
                else 0
            )
            if match > version_match[2]:
                for release_asset in release.get_assets():
                    fn, ext = os.path.splitext(release_asset.name)
                    if ext == ".ckpt":
                        version_match = (
                            os.path.join(
                                cache_dir,
                                f"{fn}_v{'_'.join(map(str, rel_version))}{ext}",
                            ),
                            release_asset.browser_download_url,
                            match,
                        )
                        break
        # Download the model weights if a matching release was found.
        if version_match[2] > 0:
            filename, url, _ = version_match
            logger.info(
                "Downloading model weights file %s from %s", filename, url
            )
            r = requests.get(url, stream=True, allow_redirects=True)
            r.raise_for_status()
            file_size = int(r.headers.get("Content-Length", 0))
            desc = "(Unknown total file size)" if file_size == 0 else ""
            r.raw.read = functools.partial(r.raw.read, decode_content=True)
            with tqdm.tqdm.wrapattr(
                r.raw, "read", total=file_size, desc=desc
            ) as r_raw, open(filename, "wb") as f:
                shutil.copyfileobj(r_raw, f)
            return filename
        else:
            logger.error(
                "No matching model weights for release v%s found, please "
                "specify your model weights explicitly using the `--model` "
                "parameter",
                __version__,
            )
            raise ValueError(
                f"No matching model weights for release v{__version__} found, "
                f"please specify your model weights explicitly using the "
                f"`--model` parameter"
            )


if __name__ == "__main__":
    main()	
