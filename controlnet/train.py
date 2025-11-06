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

controlnet_path = '/home/dell/AirExo-2/dependencies/ControlNet/'
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
    resume_path = to_absolute_path(cfg.model.resume_path)

    model = create_model(yaml_path).cpu()
    sd = None
    try:
        sd = load_state_dict(resume_path, location='cpu')
    except Exception as e:
        print(f"[setup_model] cldm.model.load_state_dict raised: {e}; will fallback to torch.load()")
    if sd is None or not isinstance(sd, dict):
        try:
            ckpt = torch.load(resume_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            else:
                sd = ckpt
        except Exception as e:
            raise RuntimeError(f"无法加载 checkpoint: {resume_path}\n错误: {e}")
    if not isinstance(sd, dict):
        raise RuntimeError(f"解析到的 checkpoint 不是 dict 类型，类型为: {type(sd)}")
    def try_strip_prefix(state_dict, prefixes=('module.', 'model.')):
        new = {}
        changed = False
        for k, v in state_dict.items():
            matched = False
            for p in prefixes:
                if k.startswith(p):
                    new_k = k[len(p):]
                    new[new_k] = v
                    matched = True
                    changed = True
                    break
            if not matched:
                new[k] = v
        return new, changed

    model_keys = set(model.state_dict().keys())
    sample_keys = list(sd.keys())[:50]
    if not any(k in model_keys for k in sample_keys):
        sd_stripped, changed = try_strip_prefix(sd)
        if changed and any(k in model_keys for k in sd_stripped.keys()):
            print("[setup_model] 检测到 checkpoint key 含前缀（module./model.），已尝试去除前缀。")
            sd = sd_stripped

    removed = []
    for k in list(sd.keys()):
        if 'position_ids' in k:
            removed.append(k)
            sd.pop(k)
    if removed:
        print("[setup_model] 已移除 checkpoint 中以下 'position_ids' 相关 keys：")
        for k in removed:
            print("  -", k)
    load_res = model.load_state_dict(sd, strict=False)
    print("[setup_model] 使用 checkpoint:", resume_path)
    print("[setup_model] load_state_dict 结果：")
    print("  missing_keys (模型需要但 checkpoint 没有)：", load_res.missing_keys)
    print("  unexpected_keys (checkpoint 有但模型不识别)：", load_res.unexpected_keys)

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

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    model = setup_model(cfg)
    callbacks = setup_callbacks(cfg)
    dataloader = setup_dataloader(cfg)
    
    output_dir = to_absolute_path(cfg.paths.output_dir)
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        devices=-1,
        accelerator="gpu",
        strategy="ddp",
        precision=cfg.training.precision,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        sync_batchnorm=True
    )
    
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
