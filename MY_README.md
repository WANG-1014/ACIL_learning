# 我的README
 
## 一、环境配置
- 1、创建虚拟环境
    ```
    conda env create -f environment.yaml
    conda activate AL
    ```
  2、项目可以在GPU上运行，也可以在CPU上运行。我的电脑显卡版本太高了，得用其它版本torch。

## 二、例程
- 1、执行老师的例程命令：
  ```(bash)
  python main.py ACIL ^
  --dataset CIFAR-100 ^
  --base-ratio 0.5 ^
  --phases 25 ^
  --data-root ~/dataset ^
  --IL-batch-size 4096 ^
  --num-workers 16 ^
  --backbone resnet32 ^
  --gamma 0.1 ^
  --buffer-size 8192 ^
  --cache-features ^
  --backbone-path ./backbones/resnet32_CIFAR-100_0.5_None
   ```
- 2、main(load_args())中传入的参数：
   ``` txt
   {
     'method': 'ACIL',
     'exp_name': 'ACIL',
     'cpu_only': False,
     'gpus': None,
     'dataset': 'CIFAR-100',
     'data_root': 'C:\\Users\\wang/dataset\\CIFAR-100',
     'num_workers': 16,
     'base_ratio': 0.5,
     'phases': 25,
     'batch_size': 256,
     'cache_features': True,
     'backbone': 'resnet32',
     'cache_path': './backbones/resnet32_CIFAR-100_0.5_None',
     'seed': None,
     'dataset_seed': None,
     'base_epochs': 300,
     'warmup_epochs': 10,
     'learning_rate': 0.5,
     'momentum': 0.9,
     'weight_decay': 0.0005,
     'separate_decay': False,
     'label_smoothing': 0.05,
     'IL_batch_size': 4096,
     'gamma': 0.1,
     'buffer_size': 8192,
     'gamma_comp': 0.1,
     'sigma': 10,
     'compensation_ratio': 1,
     'backbone_path': './backbones/resnet32_CIFAR-100_0.5_None\\backbone.pth',
     'saving_root': 'saved_models\\resnet32_CIFAR-100_0.5_None\\ACIL\\2026-05-17T14-08-26',
     'argv': "['main.py', 'ACIL', '--dataset', 'CIFAR-100', '--base-ratio', '0.5', '--phases', '25', '--data-root', '~/dataset', '--IL-batch-size', '4096', '--num-workers', '16', '--backbone', 'resnet32', '--gamma', '0.1', '--buffer-size', '8192', '--cache-features', '--backbone-path', './backbones/resnet32_CIFAR-100_0.5_None']"
   }
   ```
  



## 三、我的代码
### 2.1 课程作业要求：
  - 要求：
    1. 您只能使用数据集的前半部分来强化骨干网络。在增量学习过程中使用该数据集会导致数据泄露，并且得出的结果也不会令人信服。
    2. 这个项目需要团队合作。你应该找一个由 3 至 5 名学生组成的小组，并一起开展工作。
    3. 将 ACIL 用作增量学习的基准。您应当使用另外一半类别来与 ACIL 进行增量学习，以便进行公平的比较。
    4. 在本次任务中，推荐使用加州大学梅尔森分校的土地利用数据集。
    5. 该模型架构推荐采用 ViT-B_16。由于本项目旨在强化骨干网络，直接使用参数更多的其他骨干网络可能会有效，但不够优雅。此外，参数较少的骨干网络训练速度更快。
    6. 需要完成一份 5 页的项目报告以及一个配有幻灯片的口头报告。 报告和幻灯片都应使用英语，报告应按照模板撰写。口头报告可以用中文。
  - 一些细节说明：
    1. 每个类别有 100 张图片，后 20 张作为测试集
    2. 训练 protocol 为 base phase 训练 11 类，后续每个 phase 训练 1 类（一共21类）
    3. oral presentation 限制在 5 分钟内
    后续结合大家的反馈补充

### 2.2 我的代码实现：
- 0、将老师给的图像数据集放到./my_dataset目录下，数据集结构如下：
  ```
  my_dataset
  ├── UCMerced_LandUse
      ├── Images
          ├── class1
              ├── img1.jpg
              ├── img2.jpg
              ...
          ├── class2
              ├── img1.jpg
              ├── img2.jpg
              ...
          ...
  ```
  
- 1、执行命令：
    ```(bash)
    python main.py ACIL ^
    --dataset UCMerced_LandUse ^
    --base-ratio 0.5238095238 ^
    --phases 10 ^
    --data-root ./my_dataset ^
    --batch-size 256 ^
    --num-workers 4 ^
    --backbone vit_b_16 ^
    --learning-rate 0.5 ^
    --label-smoothing 0.05 ^
    --base-epochs 300 ^
    --weight-decay 5e-4 ^
    --gamma 0.1 ^
    --buffer-size 2048 ^
    --cache-features ^
    --IL-batch-size 64 ^
     ```
  - 注意： 
    1. --dataset UCMerced_LandUse(需要修改为自己的数据集)
    2. --base-ratio 0.5238095238(需要修改为 -> 11/20 ≈ 0.5238095238)
    3. --phase 10(需要修改为10次增量学习阶段) 
    4. --backbone vit_b_16(需要修改为vit_b_16模型架构)
    5. --seed 520(设置随机数种子，保证结果可复现)
    6. --dataset-seed 520(设置随机数种子，保证结果可复现)
    7. --num-workers 4(根据自己的计算资源调整数据加载的线程数、window运行设置低一点？)
    8. --IL-batch-size 64(根据自己的计算资源调整增量学习阶段的批量大小，我的只有8GB显存，设置为64)
    9. --buffer-size 2048(根据自己的计算资源调整缓冲区大小，我的只有8GB显存，设置为2048)

- 2、UCMerced.py
  - 写了一个UCMerced_LandUse_类用于目标数据集，继承DatasetWrapper，实际上是一个数据集包装器，主要功能是加载数据集并进行预处理。
  - 主要功能：
    - _subset、subset_at_phase、subset_until_phase：根据情况返回数据集的子集，确保每个阶段使用特定类别的数据。
    - basic_transform：基本的图像预处理方法，包括调整大小、中心裁剪和归一化等。（做测试集时采用）
    - augment_transform：数据增强方法，包含随机裁剪、水平翻转和颜色抖动等。（做训练集时采用）



TODO: 还需修改UCMerced.py的标准差和均值