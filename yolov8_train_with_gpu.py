{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyMYU6UGbxFirsP6IBRJ4Mee",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/urozor/machinelearning/blob/main/yolov8_train_with_gpu.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uXKT9yw6C75-",
        "outputId": "546d72cd-a7d8-4b66-951e-42774b794e21"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting ultralytics\n",
            "  Downloading ultralytics-8.0.227-py3-none-any.whl (660 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m660.5/660.5 kB\u001b[0m \u001b[31m7.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (3.7.1)\n",
            "Requirement already satisfied: numpy>=1.22.2 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.23.5)\n",
            "Requirement already satisfied: opencv-python>=4.6.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.8.0.76)\n",
            "Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.4.0)\n",
            "Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (6.0.1)\n",
            "Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.31.0)\n",
            "Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.11.4)\n",
            "Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.1.0+cu121)\n",
            "Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.16.0+cu121)\n",
            "Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.66.1)\n",
            "Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.5.3)\n",
            "Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.12.2)\n",
            "Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from ultralytics) (5.9.5)\n",
            "Requirement already satisfied: py-cpuinfo in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.0.0)\n",
            "Collecting thop>=0.1.1 (from ultralytics)\n",
            "  Downloading thop-0.1.1.post2209072238-py3-none-any.whl (15 kB)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.2.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (4.46.0)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.4.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (23.2)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (3.1.1)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->ultralytics) (2023.3.post1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2023.11.17)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.13.1)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (4.5.0)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.2.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.1.2)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2023.6.0)\n",
            "Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2.1.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.16.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.8.0->ultralytics) (2.1.3)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.8.0->ultralytics) (1.3.0)\n",
            "Installing collected packages: thop, ultralytics\n",
            "Successfully installed thop-0.1.1.post2209072238 ultralytics-8.0.227\n"
          ]
        }
      ],
      "source": [
        "!pip install ultralytics"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from ultralytics import YOLO\n",
        "# Load a model\n",
        "#model = YOLO('yolov8n.yaml')  # build a new model from YAML\n",
        "#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)\n",
        "model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights\n",
        "\n",
        "# Train the model\n",
        "results = model.train(data='coco128.yaml', device=0, epochs=100, imgsz=1280)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kNt3jvu_Gl3-",
        "outputId": "b53e8703-aae7-4646-c058-5aa4eade06a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "                   from  n    params  module                                       arguments                     \n",
            "  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 \n",
            "  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                \n",
            "  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             \n",
            "  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                \n",
            "  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             \n",
            "  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               \n",
            "  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           \n",
            "  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              \n",
            "  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           \n",
            "  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 \n",
            " 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 \n",
            " 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  \n",
            " 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            " 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 \n",
            " 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            " 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 \n",
            " 22        [15, 18, 21]  1    897664  ultralytics.nn.modules.head.Detect           [80, [64, 128, 256]]          \n",
            "YOLOv8n summary: 225 layers, 3157200 parameters, 3157184 gradients, 8.9 GFLOPs\n",
            "\n",
            "Transferred 355/355 items from pretrained weights\n",
            "Ultralytics YOLOv8.0.227 🚀 Python-3.10.12 torch-2.1.0+cu121 CUDA:0 (Tesla T4, 15102MiB)\n",
            "\u001b[34m\u001b[1mengine/trainer: \u001b[0mtask=detect, mode=train, model=yolov8n.yaml, data=coco128.yaml, epochs=100, patience=50, batch=16, imgsz=1280, save=True, save_period=-1, cache=False, device=0, workers=8, project=None, name=train2, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train2\n",
            "\n",
            "                   from  n    params  module                                       arguments                     \n",
            "  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 \n",
            "  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                \n",
            "  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             \n",
            "  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                \n",
            "  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             \n",
            "  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               \n",
            "  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           \n",
            "  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              \n",
            "  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           \n",
            "  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 \n",
            " 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 \n",
            " 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  \n",
            " 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            " 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 \n",
            " 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            " 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 \n",
            " 22        [15, 18, 21]  1    897664  ultralytics.nn.modules.head.Detect           [80, [64, 128, 256]]          \n",
            "YOLOv8n summary: 225 layers, 3157200 parameters, 3157184 gradients, 8.9 GFLOPs\n",
            "\n",
            "Transferred 355/355 items from pretrained weights\n",
            "\u001b[34m\u001b[1mTensorBoard: \u001b[0mStart with 'tensorboard --logdir runs/detect/train2', view at http://localhost:6006/\n",
            "Freezing layer 'model.22.dfl.conv.weight'\n",
            "\u001b[34m\u001b[1mAMP: \u001b[0mrunning Automatic Mixed Precision (AMP) checks with YOLOv8n...\n",
            "\u001b[34m\u001b[1mAMP: \u001b[0mchecks passed ✅\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mtrain: \u001b[0mScanning /content/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 [00:00<?, ?it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n",
            "\u001b[34m\u001b[1mval: \u001b[0mScanning /content/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 [00:00<?, ?it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Plotting labels to runs/detect/train2/labels.jpg... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m AdamW(lr=0.000119, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py:2443: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
            "  if size_prods == 1:\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Image sizes 1280 train, 1280 val\n",
            "Using 2 dataloader workers\n",
            "Logging results to \u001b[1mruns/detect/train2\u001b[0m\n",
            "Starting training for 100 epochs...\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      1/100      11.2G      1.503      2.265      1.666        228       1280: 100%|██████████| 8/8 [00:10<00:00,  1.36s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.08s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.385       0.43      0.438      0.291\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      2/100      11.3G      1.406      1.843      1.524        152       1280: 100%|██████████| 8/8 [00:06<00:00,  1.32it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.18s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.53      0.418      0.476      0.312\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      3/100      10.3G      1.333      1.664      1.501        164       1280: 100%|██████████| 8/8 [00:05<00:00,  1.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.30s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.566       0.43      0.491      0.321\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      4/100      10.8G      1.314      1.643      1.469        234       1280: 100%|██████████| 8/8 [00:05<00:00,  1.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.22s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.581      0.449      0.516      0.338\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      5/100      10.7G       1.32      1.564      1.421        249       1280: 100%|██████████| 8/8 [00:05<00:00,  1.35it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.01it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.619      0.467       0.55      0.351\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      6/100      10.3G      1.284      1.477      1.389        229       1280: 100%|██████████| 8/8 [00:05<00:00,  1.36it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.06s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.585      0.512      0.575      0.372\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      7/100      10.8G      1.237      1.441      1.401        252       1280: 100%|██████████| 8/8 [00:06<00:00,  1.19it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.07it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.614      0.513       0.59      0.389\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      8/100      10.4G      1.272      1.441      1.412        132       1280: 100%|██████████| 8/8 [00:06<00:00,  1.22it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.01it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.656      0.535        0.6      0.401\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "      9/100      10.8G      1.257      1.392      1.394        199       1280: 100%|██████████| 8/8 [00:05<00:00,  1.47it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.15s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.665      0.553      0.621      0.414\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     10/100      10.9G      1.308      1.422      1.445        199       1280: 100%|██████████| 8/8 [00:05<00:00,  1.45it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.26s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.677      0.571       0.65      0.433\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     11/100      11.1G      1.242      1.345      1.394        136       1280: 100%|██████████| 8/8 [00:05<00:00,  1.37it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.20s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.726      0.563      0.672      0.452\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     12/100        11G      1.289      1.436       1.44        171       1280: 100%|██████████| 8/8 [00:05<00:00,  1.40it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.11it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.727      0.573      0.687      0.466\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     13/100      11.6G      1.229      1.307      1.392        253       1280: 100%|██████████| 8/8 [00:05<00:00,  1.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.06it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.731      0.619      0.706      0.481\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     14/100        11G      1.177       1.24      1.331        203       1280: 100%|██████████| 8/8 [00:06<00:00,  1.18it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.03it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.702      0.662      0.724      0.491\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     15/100      10.5G      1.192      1.236      1.378        212       1280: 100%|██████████| 8/8 [00:06<00:00,  1.17it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.01it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.767      0.633      0.734      0.499\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     16/100      12.3G      1.148      1.207      1.326        199       1280: 100%|██████████| 8/8 [00:05<00:00,  1.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.12s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.755      0.648      0.741       0.51\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     17/100      10.7G      1.091      1.165      1.322        262       1280: 100%|██████████| 8/8 [00:05<00:00,  1.46it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.23s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.782      0.647       0.76      0.524\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     18/100      10.4G      1.158      1.157      1.341        253       1280: 100%|██████████| 8/8 [00:05<00:00,  1.37it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.01s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.688      0.708      0.768      0.537\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     19/100      10.7G      1.172      1.199       1.34        264       1280: 100%|██████████| 8/8 [00:05<00:00,  1.36it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.02it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.73        0.7      0.775      0.547\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     20/100      10.7G      1.118      1.107      1.309        184       1280: 100%|██████████| 8/8 [00:06<00:00,  1.22it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.15s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.77      0.686      0.794      0.557\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     21/100      10.9G      1.111      1.097      1.299        262       1280: 100%|██████████| 8/8 [00:07<00:00,  1.11it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.10it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.773      0.694      0.808       0.57\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     22/100      11.7G      1.108      1.054      1.306        213       1280: 100%|██████████| 8/8 [00:06<00:00,  1.23it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.01it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.759      0.719      0.815       0.58\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     23/100      10.4G      1.075      1.036      1.282        235       1280: 100%|██████████| 8/8 [00:05<00:00,  1.51it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.42s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.752      0.732      0.822       0.59\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     24/100      10.8G      1.033      1.017      1.263        211       1280: 100%|██████████| 8/8 [00:05<00:00,  1.52it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.01it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.767      0.745      0.837      0.594\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     25/100      10.3G      1.088      1.096      1.262        237       1280: 100%|██████████| 8/8 [00:05<00:00,  1.48it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.04it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.784      0.733      0.829      0.595\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     26/100      10.8G      1.045      1.002      1.249        194       1280: 100%|██████████| 8/8 [00:08<00:00,  1.05s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.33s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.79      0.733      0.845      0.603\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     27/100      10.7G      1.104      1.066      1.294        186       1280: 100%|██████████| 8/8 [00:06<00:00,  1.15it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00,  1.91s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.756      0.766       0.85      0.606\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     28/100      10.3G      1.013     0.9728      1.232        251       1280: 100%|██████████| 8/8 [00:07<00:00,  1.13it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.02it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.769      0.778      0.848      0.613\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     29/100      10.9G      1.025          1      1.248        186       1280: 100%|██████████| 8/8 [00:06<00:00,  1.15it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.12it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.812      0.758      0.852      0.618\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     30/100      10.8G      1.076      1.026        1.3        186       1280: 100%|██████████| 8/8 [00:07<00:00,  1.05it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.30s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.81      0.772      0.863      0.628\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     31/100      10.5G      1.044     0.9772      1.246        240       1280: 100%|██████████| 8/8 [00:06<00:00,  1.20it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.00it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.849      0.754      0.861      0.627\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     32/100      10.8G      1.074      1.023      1.276        216       1280: 100%|██████████| 8/8 [00:06<00:00,  1.15it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.23s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.828      0.784      0.871      0.634\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     33/100      10.2G      1.001     0.9596      1.246        244       1280: 100%|██████████| 8/8 [00:08<00:00,  1.01s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.06s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.838      0.779      0.868      0.631\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     34/100      11.2G      1.013     0.9299      1.226        214       1280: 100%|██████████| 8/8 [00:07<00:00,  1.04it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.02s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.846      0.808       0.88      0.633\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     35/100      10.8G     0.9697      0.904      1.223        194       1280: 100%|██████████| 8/8 [00:07<00:00,  1.01it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:06<00:00,  1.57s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.851      0.819      0.886      0.639\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     36/100      10.9G     0.9976     0.9216      1.236        196       1280: 100%|██████████| 8/8 [00:05<00:00,  1.38it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.09s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.835       0.83      0.897      0.648\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     37/100      11.1G     0.9492     0.8744      1.187        260       1280: 100%|██████████| 8/8 [00:06<00:00,  1.18it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.02s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.856      0.817      0.898      0.649\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     38/100      10.3G      1.021     0.9367      1.258        244       1280: 100%|██████████| 8/8 [00:08<00:00,  1.04s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.05s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.856      0.819      0.898      0.652\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     39/100      10.8G      0.983     0.9046      1.209        301       1280: 100%|██████████| 8/8 [00:06<00:00,  1.31it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.08s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.833      0.846      0.898      0.657\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     40/100      10.3G     0.9824     0.9166       1.22        186       1280: 100%|██████████| 8/8 [00:07<00:00,  1.09it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.04s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.849      0.821      0.902      0.663\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     41/100      10.3G      0.935     0.8512      1.185        241       1280: 100%|██████████| 8/8 [00:07<00:00,  1.14it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.35s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.858      0.808      0.895      0.661\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     42/100      10.2G     0.9786     0.9008       1.23        218       1280: 100%|██████████| 8/8 [00:05<00:00,  1.39it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.33s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.87      0.807      0.897      0.666\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     43/100      10.9G     0.9576     0.8507      1.205        351       1280: 100%|██████████| 8/8 [00:05<00:00,  1.36it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.07it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.879      0.817      0.903      0.675\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     44/100      11.7G       0.98     0.8715      1.208        139       1280: 100%|██████████| 8/8 [00:06<00:00,  1.18it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.02s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.899      0.813       0.92      0.677\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     45/100      10.3G       0.96      0.851      1.225        262       1280: 100%|██████████| 8/8 [00:08<00:00,  1.04s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.08s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.841      0.867      0.924      0.681\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     46/100      10.9G       0.97     0.8604      1.188        201       1280: 100%|██████████| 8/8 [00:06<00:00,  1.17it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.01it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.85      0.864      0.925      0.685\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     47/100      10.9G     0.9569      0.868      1.199        223       1280: 100%|██████████| 8/8 [00:07<00:00,  1.13it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.26s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.857       0.86      0.923      0.687\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     48/100      11.6G     0.9521     0.8318      1.177        268       1280: 100%|██████████| 8/8 [00:06<00:00,  1.15it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.01it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.894      0.821      0.922      0.687\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     49/100      10.6G      0.959     0.8433      1.207        221       1280: 100%|██████████| 8/8 [00:06<00:00,  1.15it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.08s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.92      0.811      0.926      0.697\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     50/100      10.7G     0.9619     0.8211       1.21        267       1280: 100%|██████████| 8/8 [00:09<00:00,  1.18s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:06<00:00,  1.52s/it]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.928      0.818       0.93      0.702\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     51/100      10.3G     0.9308     0.8537       1.22        186       1280: 100%|██████████| 8/8 [00:07<00:00,  1.02it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.05it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.911      0.833      0.931      0.704\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     52/100      10.4G     0.9602     0.8806      1.203        224       1280: 100%|██████████| 8/8 [00:08<00:00,  1.05s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.04it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.872      0.855      0.929        0.7\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     53/100      10.6G     0.9482      0.828      1.186        213       1280: 100%|██████████| 8/8 [00:08<00:00,  1.03s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.35s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.871      0.875       0.93      0.702\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     54/100      10.5G     0.9942     0.8813      1.206        143       1280: 100%|██████████| 8/8 [00:09<00:00,  1.15s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.27s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.886      0.845      0.923      0.702\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     55/100      11.4G     0.9451     0.8381      1.175        180       1280: 100%|██████████| 8/8 [00:05<00:00,  1.38it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.01s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.882      0.862      0.934      0.704\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     56/100      10.2G     0.9231     0.8056      1.157        241       1280: 100%|██████████| 8/8 [00:06<00:00,  1.30it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.01s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.865      0.873      0.935      0.706\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     57/100      10.6G     0.9196     0.8193       1.17        230       1280: 100%|██████████| 8/8 [00:07<00:00,  1.01it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.33s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.86      0.878      0.935      0.711\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     58/100      10.1G     0.8882     0.7941      1.169        208       1280: 100%|██████████| 8/8 [00:07<00:00,  1.12it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.87      0.875      0.936       0.71\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     59/100      11.3G     0.9163     0.7947       1.17        247       1280: 100%|██████████| 8/8 [00:06<00:00,  1.31it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.16s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.865      0.874      0.938      0.716\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     60/100      11.1G     0.8896     0.8008      1.166        215       1280: 100%|██████████| 8/8 [00:07<00:00,  1.03it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.23s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.874      0.876      0.939      0.715\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     61/100      10.8G     0.9315     0.7768      1.164        193       1280: 100%|██████████| 8/8 [00:06<00:00,  1.31it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.07s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.87      0.876      0.939      0.717\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     62/100      10.6G     0.8979     0.8161      1.171        224       1280: 100%|██████████| 8/8 [00:09<00:00,  1.15s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.10s/it]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.867      0.881      0.939      0.716\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     63/100      10.1G     0.9234     0.8072      1.185        158       1280: 100%|██████████| 8/8 [00:06<00:00,  1.19it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.07it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.875      0.879      0.941      0.719\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     64/100      10.1G      0.954       0.85      1.186        240       1280: 100%|██████████| 8/8 [00:07<00:00,  1.11it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.07it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.898       0.86      0.941      0.717\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     65/100      11.5G     0.9108     0.8044       1.15        328       1280: 100%|██████████| 8/8 [00:05<00:00,  1.48it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.42s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.897      0.866      0.942       0.72\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     66/100      10.1G     0.8954     0.7986       1.15        197       1280: 100%|██████████| 8/8 [00:05<00:00,  1.43it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.11s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.906      0.864      0.943       0.72\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     67/100      10.5G     0.9098     0.8217      1.172        219       1280: 100%|██████████| 8/8 [00:06<00:00,  1.15it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.24s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.91       0.86      0.944      0.723\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     68/100      10.3G      0.901      0.789      1.162        186       1280: 100%|██████████| 8/8 [00:07<00:00,  1.08it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.33s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.871      0.885      0.945      0.725\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     69/100      10.8G      0.846     0.7554      1.154        232       1280: 100%|██████████| 8/8 [00:06<00:00,  1.17it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.16s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929       0.86      0.896      0.946      0.731\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     70/100      10.9G     0.8843     0.7692      1.147        219       1280: 100%|██████████| 8/8 [00:07<00:00,  1.11it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:06<00:00,  1.65s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.906      0.847      0.945      0.728\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     71/100      10.8G     0.8943       0.78      1.164        222       1280: 100%|██████████| 8/8 [00:05<00:00,  1.41it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.05s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.918      0.842      0.945      0.729\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     72/100      10.1G     0.8701     0.7401       1.13        236       1280: 100%|██████████| 8/8 [00:06<00:00,  1.23it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.13it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.907      0.851      0.945       0.73\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     73/100      10.6G     0.9042     0.7732      1.167        214       1280: 100%|██████████| 8/8 [00:05<00:00,  1.47it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:06<00:00,  1.55s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.918      0.843      0.946      0.731\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     74/100      10.7G     0.8812     0.7519      1.145        251       1280: 100%|██████████| 8/8 [00:07<00:00,  1.01it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.02s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.913      0.851      0.946      0.735\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     75/100      10.3G     0.9009     0.8137      1.185        226       1280: 100%|██████████| 8/8 [00:06<00:00,  1.25it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.21s/it]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.917      0.848      0.947      0.737\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     76/100      10.2G     0.8968     0.7736      1.133        231       1280: 100%|██████████| 8/8 [00:06<00:00,  1.21it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.09it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.909      0.855      0.949      0.737\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     77/100      10.7G     0.8992     0.7596      1.165        224       1280: 100%|██████████| 8/8 [00:06<00:00,  1.16it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.01s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.911      0.855      0.948      0.735\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     78/100      10.3G     0.8727     0.7664      1.153        202       1280: 100%|██████████| 8/8 [00:07<00:00,  1.02it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.13s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.906      0.859      0.949      0.735\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     79/100      10.5G     0.8835     0.8105      1.177        176       1280: 100%|██████████| 8/8 [00:09<00:00,  1.14s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,  1.18s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.872      0.896      0.949      0.733\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     80/100      10.9G       0.84     0.7327      1.113        236       1280: 100%|██████████| 8/8 [00:06<00:00,  1.16it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00,  1.87s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.878        0.9      0.951      0.736\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     81/100      10.5G      0.864     0.7391      1.145        267       1280: 100%|██████████| 8/8 [00:08<00:00,  1.10s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.02it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.874      0.903       0.95      0.737\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     82/100      10.3G     0.8308     0.7217      1.127        219       1280: 100%|██████████| 8/8 [00:07<00:00,  1.08it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:03<00:00,  1.03it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.878      0.903       0.95      0.738\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     83/100      10.3G     0.8546     0.7351      1.128        182       1280: 100%|██████████| 8/8 [00:13<00:00,  1.64s/it]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00,  1.46s/it]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                   all        128        929      0.882      0.888       0.95      0.742\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "     84/100      11.5G     0.8627     0.7606      1.138        258       1280: 100%|██████████| 8/8 [00:05<00:00,  1.38it/s]\n",
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  50%|█████     | 2/4 [00:01<00:01,  1.32it/s]"
          ]
        }
      ]
    }
  ]
}