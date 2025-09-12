# AirExo-2-Test 项目说明


## 主要 Python 脚本功能简述

- `double_arm.py`：加载完整机械臂 URDF，支持双臂模型的运动学与几何建模，含数据读取与处理。
- `double_arm_withoutbody.py`：加载并过滤仅包含上半身（左右机械臂和夹爪）的模型，适合仿真和部分运动学分析。
- `inv_new.py` / `inv_work_move.py`：空文件，预留或历史测试用。
- `inv_save.py` / `inv_work.py` / `inv_work_save.py`：加载机械臂模型，支持逆运动学、数据处理与保存，适合批量仿真与分析。
- `mirror.py`：对视频进行左右镜像处理，依赖 moviepy。
- `single_urdf.py`：加载单臂及夹爪模型，适合单臂仿真与运动学分析。
- `singlearm.py`：加载并过滤仅包含上半身的单臂模型，适合单臂仿真。
- `test_rm.py`：调用 Robotic_Arm 库，连接机械臂硬件并获取软件信息。

## 环境依赖对比

### requirements.txt 主要依赖
- numpy==1.24.3
- pillow==10.3.0
- colorlog==6.8.2
- termcolor==2.4.0
- tqdm==4.66.2
- kinpy==0.2.2
- ipython==8.12.3
- h5py==3.11.0
- open3d==0.16.0
- hydra-core==1.3.2
- pyserial==3.5
- omegaconf==2.3.0
- pyrealsense2==2.55.1.6486
- opencv-python==4.5.5.64
- opencv-contrib-python==4.5.5.64
- torch==1.13.0
- torchvision==0.14.0
- torchaudio==0.13.0
- transformers==4.46.3
- einops==0.3.0
- open_clip_torch==2.0.2
- pytorch-lightning==1.5.0
- pytorch-kinematics==0.7.5
- pynput==1.7.7
- six==1.17.0
- imageio==2.9.0
- scipy==1.10.1
- matplotlib==3.7.5
- natsort==8.4.0
- imageio-ffmpeg==0.4.2

### ik_airexo 虚拟环境主要差异
- numpy 版本为 2.0.2（高于 requirements.txt）
- imageio 版本为 2.37.0（高于 requirements.txt）
- imageio-ffmpeg 版本为 0.6.0（高于 requirements.txt）
- opencv-python 版本为 4.12.0.88（高于 requirements.txt）
- 额外包含 moviepy、matplotlib-inline、meshcat、pandas、scikit-learn、transformations、Robotic_Arm 等包
- 依赖包数量更多，部分为硬件或仿真相关扩展

建议：如遇依赖冲突，优先以虚拟环境实际包为准，或根据 requirements.txt 进行适当调整。

## 环境安装教程

### 1. 创建并激活 conda 虚拟环境
```bash
conda create -n ik_airexo python=3.9
conda activate ik_airexo
```

### 2. 安装依赖包
推荐直接使用 requirements.txt 安装：
```bash
pip install -r requirements.txt
```
如需额外包（如 moviepy、meshcat、matplotlib-inline、pandas、scikit-learn、transformations、Robotic_Arm），可单独安装：
```bash
pip install moviepy meshcat matplotlib-inline pandas scikit-learn transformations Robotic_Arm
```

### 3. 安装 pinocchio（如有需要）
建议使用 conda-forge 安装，兼容性更好：
```bash
conda install -c conda-forge pinocchio
```

### 4. 运行示例
以 `double_arm.py` 为例：
```
python double_arm.py
```

## 操作流程详解

也可以查看hand_readme.html文件

### 1. 视频帧分割
使用 ffmpeg 将视频分割为图像帧：
- 从 1 开始编号：
```bash
ffmpeg -i input_video.mp4 -vf "fps=30" /home/tracy/airexo/AirExo-2/data/train/scene_0036/color/%d.png
```
- 从 0 开始编号：
```bash
ffmpeg -start_number 0 -i input_video.mp4 -vf "fps=30" /home/tracy/airexo/AirExo-2/data/train/scene_0036/color/%d.png
```
> 注意：输出文件夹只包含提取的图像，不要混入原始视频。

### 2. SAM2 批量处理流程
1. 进入 `/utils/sam2/` 目录，运行 `main.py`。
2. 输入图像帧总数，稍等后显示首帧。
3. 点击希望移除的区域，绿色框自动出现。
4. 按 Enter 或关闭窗口，自动处理所有帧。
5. 配置输出路径于 `default.yaml`，可设置 `start_frame: 0` 或 `start_frame: 1`。

### 3. Propainter 视频修复
1. 运行 `/airexo/adaptor/inpainting.py`，终端会输出最终视频路径。
2. 参数配置在 `/airexo/configs/inpainting.yaml`，如：
```yaml
video:
  resize_ratio: 0.5
processing:
  neighbor_length: 5
  subvideo_length: 40
runtime:
  fp16: true
```

### 4. ControlNet 训练与推理
- 推理脚本：`/airexo/adaptor/controlnet_inference.py`
- 训练脚本：`/utils/controlnet/train.py`

训练前需设置代理环境变量：
```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="socks5://127.0.0.1:7890"
curl -I https://huggingface.co
```
验证网络连接后即可训练。

### 5. Robot 配置与视角调整
#### 配置流程
1. 替换 `airexo/urdf_models` 文件夹。
2. 修改 `configs/tests/urdf/robot.yaml` 的 `urdf_file` 路径为 `airexo/urdf_models/robot/true_robot.urdf`。
3. 替换 `airexo/tests/urdf_robot.py` 内容为官方示例（见 hand_readme.html 代码段）。
4. 进入虚拟环境后运行：
```bash
python -m airexo.tests.urdf_robot
```

#### 视角调整
1. 安装依赖：
```bash
pip install moviepy imageio[ffmpeg]
pip install --upgrade imageio
pip install --upgrade imageio-ffmpeg
```

2. 视频镜像处理：
```
python mirror.py
```