# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import os

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
import argparse
import logging
import platform
from pathlib import Path

import torch
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # WandbLogger
from pytorch_lightning.strategies import DDPStrategy

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")
from collections import OrderedDict

from AR.utils import get_newest_ckpt
from process_ckpt import my_save


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        # if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if (
                    self.if_save_latest == True
                ):  ####如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)
                    # torch.save(
                    # print(os.environ)
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        my_save(
                            to_save_od,
                            "%s/%s-e%s.ckpt"
                            % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)

def run_args(name, prepare_dir, output_dir, is_half):
    config = {
        "data": {
            "max_eval_sample": 8,
            "max_sec": 54,
            "num_workers": 4,
            "pad_val": 1024
        },
        "inference": {
            "top_k": 15
        },
        "model": {
            "EOS": 1024,
            "dropout": 0,
            "embedding_dim": 512,
            "head": 16,
            "hidden_dim": 512,
            "linear_units": 2048,
            "n_layer": 24,
            "phoneme_vocab_size": 732,
            "random_bert": 0,
            "vocab_size": 1025
        },
        "optimizer": {
            "decay_steps": 40000,
            "lr": 0.01,
            "lr_end": 0.0001,
            "lr_init": 0.00001,
            "warmup_steps": 2000
        },
        "output_dir": "logs/test/logs_s1_v2Pro",
        "pretrained_s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "train": {
            "batch_size": 2,
            "epochs": 15,
            "exp_name": "test",
            "gradient_clip": 1,
            "half_weights_save_dir": "GPT_weights_v2Pro",
            "if_dpo": False,
            "if_save_every_weights": True,
            "if_save_latest": True,
            "precision": "32",
            "save_every_n_epoch": 5,
            "seed": 1234
        },
        "train_phoneme_path": "logs/test/2-name2text.txt",
        "train_semantic_path": "logs/test/6-name2semantic.tsv"
    }

    # 每张显卡的batch_size
    batch_size = 4
    if is_half == False:
        config["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)
    config["train"]["batch_size"] = batch_size
    # 总训练轮数total_epoch
    config["train"]["epochs"] = 15
    # config["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    # 保存频率save_every_epoch
    # config["train"]["save_every_n_epoch"] = 5
    # config["train"]["if_save_every_weights"] = if_save_every_weights
    # config["train"]["if_save_latest"] = if_save_latest
    # 是否开启DPO训练选项(实验性)
    # config["train"]["if_dpo"] = if_dpo
    config["train"]["half_weights_save_dir"] = os.path.join(output_dir, 'gpt_train')
    os.makedirs(config["train"]["half_weights_save_dir"], exist_ok=True)
    config["train"]["exp_name"] = name
    config["train_semantic_path"] = "%s/6-name2semantic.tsv" % prepare_dir
    config["train_phoneme_path"] = "%s/2-name2text.txt" % prepare_dir
    config["output_dir"] = "%s/logs_s1_%s" % (prepare_dir, 'v2Pro')

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # val_check_interval=9999999999999999999999,###不要验证
        # check_val_every_n_epoch=None,
        limit_val_batches=0,
        devices=-1 if torch.cuda.is_available() else 1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
        if torch.cuda.is_available()
        else "auto",
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,  # 非常简单的修改，但解决了采用自定义的 bucket_sampler 下训练步数不一致的问题！
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
        # dev_semantic_path=args.dev_semantic_path,
        # dev_phoneme_path=args.dev_phoneme_path
    )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

def main(args):
    config = load_yaml_config(args.config_file)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # val_check_interval=9999999999999999999999,###不要验证
        # check_val_every_n_epoch=None,
        limit_val_batches=0,
        devices=-1 if torch.cuda.is_available() else 1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
        if torch.cuda.is_available()
        else "auto",
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,  # 非常简单的修改，但解决了采用自定义的 bucket_sampler 下训练步数不一致的问题！
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
        # dev_semantic_path=args.dev_semantic_path,
        # dev_phoneme_path=args.dev_phoneme_path
    )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1longer.yaml",
        help="path of config file",
    )
    # args for dataset
    # parser.add_argument('--train_semantic_path',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/6-name2semantic.tsv')
    # parser.add_argument('--train_phoneme_path', type=str, default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/2-name2text.txt')

    # parser.add_argument('--dev_semantic_path', type=str, default='dump_mix/semantic_dev.tsv')
    # parser.add_argument('--dev_phoneme_path', type=str, default='dump_mix/phoneme_dev.npy')
    # parser.add_argument('--output_dir',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/logs_s1',help='directory to save the results')
    # parser.add_argument('--output_dir',type=str,default='/liujing04/gpt_logs/s1/xuangou_ft',help='directory to save the results')

    args = parser.parse_args()
    logging.info(str(args))
    main(args)
