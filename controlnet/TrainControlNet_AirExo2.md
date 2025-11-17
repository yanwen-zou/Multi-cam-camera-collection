# Train ControlNet (AirExo-2)

## 1. 环境准备


1.  **创建并激活 Conda 环境**
    该项目依赖 Python 3.8

    ```bash
    conda create -n airexo3.8 python=3.8
    conda activate airexo3.8
    ```

2.  **安装依赖包**
    在仓库的根目录执行以下命令，安装 `requirements.txt` （airexo）中列出的所有依赖：

    ```bash
    pip install -r requirements.txt
    ```

## 2. 准备预训练模型与下载controlnet源码

ControlNet 的训练需要一个预训练的 Stable Diffusion 权重作为基础。

1. **下载模型**
   请从以下链接下载 `control_sd15_ini.ckpt` 文件：
   [https://huggingface.co/Boese0601/MagicDance/blob/main/control_sd15_ini.ckpt](https://huggingface.co/Boese0601/MagicDance/blob/main/control_sd15_ini.ckpt)

2. **存放模型**
   建议在项目根目录下创建一个 `models` 文件夹，并将下载的 `.ckpt` 文件放入其中，路径如下：
   `./models/control_sd15_ini.ckpt`

   (如果您选择存放在其他位置，请确保在步骤 4 的配置文件中正确指向该路径。)

3. **controlnet源码**

   请从以下链接下载controlnet源码：

   https://github.com/lllyasviel/ControlNet
   并放置在dependencies文件夹下。

## 3. 准备数据集

训练数据需要包含仿真图片 (rgb) 和对应的条件图 (color，即“真机”图)。

1.  **数据结构**
    请按照以下结构组织您的数据集：

    ```
    /data/
    ├── rgb/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    └── color/
        ├── 0001.png
        ├── 0002.png
        └── ...
    ```

2.  **重要提示**
    * `rgb` 文件夹存放原始训练图片。
    * `color` 文件夹存放对应的条件控制图。
    * 两个文件夹中的图片必须**一一对应**，且**文件名完全相同**（包括扩展名），以便训练脚本正确配对。

## 4. 配置文件

所有训练参数（如学习率、批次大小、模型路径、数据路径等）都在 `config/train.yaml` 配置文件中定义。

在开始训练之前，请务必**打开并修改** `train.yaml` 您将要使用的配置文件：

```bash
paths:
  camera_dir: "/data/color"  # Directory containing camera images
  render_dir: "/data/rgb"  # Directory containing rendered images
  output_dir: "/checkpoint"  # Directory to save checkpoints

model:
  yaml_path: null  # Path to model yaml file (empty uses default)
  resume_path: "control_sd15_ini.ckpt" # Path to checkpoint to resume from
  sd_locked: true  # Whether to lock stable diffusion model weights
  only_mid_control: false  # Whether to only use mid-level control

training:
  prompt: "robotic arms, dual arm, industrial robotic manipulator, metallic silver color, mechanical joints, precise mechanical details, gripper end effector, high quality photo, photorealistic, clear and sharp details" # Text prompt for conditioning
  batch_size: 22  # Batch size for training
  num_workers: 4  # Number of workers for data loading
  learning_rate: 1e-5  # Learning rate
  precision: 32  # Training precision (16 or 32)
  img_size: 512  # Size to resize images to
  logger_freq: 300  # Frequency of image logging
  save_steps: 1000  # Save checkpoint every N steps
  accumulate_grad_batches: 4  # Gradient accumulation steps
```

## 5. 开始训练

当您的环境、模型、数据和配置都准备就绪后，运行以下命令开始训练：

```bash
python utils/controlnet/train.py
```

**注意，上面所有步骤都已经在4090上部署好，只需要进入conda环境和仓库目录，将真机和仿真图片放入对应的文件夹，运行python utils/controlnet/train.py即可开始训练。**