import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ['NCCL_P2P_DISABLE'] = '1'

controlnet_path = '/home/ryan/Documents/GitHub/AirExo-2-test/dependencies/ControlNet/'
sys.path.append(controlnet_path)

from share import *
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from dataset import RoboticArmDataset

def setup_model(cfg):
    yaml_path = cfg.model.yaml_path
    if not yaml_path:
        yaml_path = os.path.join(controlnet_path, 'models/cldm_v15.yaml')
    else:
        yaml_path = to_absolute_path(yaml_path)
    
    print(f"使用配置文件: {yaml_path}")
    
    # 创建模型
    model = create_model(yaml_path).cpu()
    
    # 检查是否指定了预训练模型路径
    if cfg.model.resume_path and cfg.model.resume_path.strip():
        resume_path = to_absolute_path(cfg.model.resume_path)
        
        if os.path.exists(resume_path):
            try:
                print(f"加载预训练模型: {resume_path}")
                model.load_state_dict(load_state_dict(resume_path, location='cpu'))
                print("成功加载预训练模型")
            except Exception as e:
                print(f"加载预训练模型失败: {e}")
                print("使用随机初始化的模型继续训练")
        else:
            print(f"预训练模型文件不存在: {resume_path}")
            print("使用随机初始化的模型开始训练")
    else:
        print("未指定预训练模型，使用随机初始化的模型开始训练")
    
    # 设置训练参数
    model.learning_rate = cfg.training.learning_rate
    model.sd_locked = cfg.model.sd_locked
    model.only_mid_control = cfg.model.only_mid_control
    
    return model

def setup_callbacks(cfg):
    output_dir = to_absolute_path(cfg.paths.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=-1,
        every_n_train_steps=cfg.training.save_steps,
    )
    
    logger = ImageLogger(batch_frequency=cfg.training.logger_freq)
    
    return [checkpoint_callback, logger]

def setup_dataloader(cfg):
    camera_dir = to_absolute_path(cfg.paths.camera_dir)
    render_dir = to_absolute_path(cfg.paths.render_dir)
    
    dataset = RoboticArmDataset(
        camera_dir=camera_dir,
        render_dir=render_dir,
        prompt=cfg.training.prompt,
        img_size=cfg.training.img_size
    )
    
    dataloader = DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    
    return dataloader

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print("开始设置模型...")
    model = setup_model(cfg)
    
    # print("设置回调函数...")
    # callbacks = setup_callbacks(cfg)
    
    # print("设置数据加载器...")
    # dataloader = setup_dataloader(cfg)
    
    # output_dir = to_absolute_path(cfg.paths.output_dir)
    # print(f"输出目录: {output_dir}")
    
    # trainer = pl.Trainer(
    #     default_root_dir=output_dir,
    #     devices=-1,
    #     accelerator="gpu",
    #     strategy="ddp",
    #     precision=cfg.training.precision,
    #     callbacks=callbacks,
    #     accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    #     sync_batchnorm=True
    # )
    
    # print("开始训练...")
    # trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()