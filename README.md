# predrnn_benchmark

## Get Started

1. Install Python 3.6, PyTorch 1.9.0 for the main code. Also, install Tensorflow 2.1.0 for BAIR dataloader.

2. Download data. This repo contains code for three datasets: the [Benchmark dataset] (https://drive.google.com/drive/folders/1U4b1iG_67-gNfHWKfLzSSRy4Ong0zv3d?usp=sharing) [Moving Mnist dataset](https://onedrive.live.com/?authkey=%21AGzXjcOlzTQw158&id=FF7F539F0073B9E2%21124&cid=FF7F539F0073B9E2), the [KTH action dataset](https://drive.google.com/drive/folders/1_M1O4TuQOhYcNdXXuNoNjYyzGrSM9pBF?usp=sharing), and the BAIR dataset (30.1GB), which can be obtained by:

   ```
   wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
   ```

3. Train the model. You can use the following bash script to train the model. The learned model will be saved in the `--save_dir` folder.
  The generated future frames will be saved in the `--gen_frm_dir` folder.

4. You can get **pretrained models** from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/72241e0046a74f81bf29/) or [Google Drive](https://drive.google.com/drive/folders/1jaEHcxo_UgvgwEWKi0ygX1SbODGz6PWw).
```
cd benchmark_script/
sh predrnn_benchmark_train.sh
sh predrnn_v2_benchmark_train.sh

cd mnist_script/
sh predrnn_mnist_train.sh
sh predrnn_v2_mnist_train.sh

cd kth_script/
sh predrnn_kth_train.sh
sh predrnn_v2_kth_train.sh

cd bair_script/
sh predrnn_bair_train.sh
sh predrnn_v2_bair_train.sh
```
