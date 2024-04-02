# 配置环境
```
conda env create -f env/env.yml
pip install -r env/requirements.txt
python setup.py develop --no_cuda_ext
```

# 下载/配置数据集
数据集路径请在`options/train/Rain13k/Wavelet-width32.yml`内修改，需要修改验证数据集合位置和训练数据集位置

## SIDD
修改`scripts/data_preparation/sidd.py`中的数据路径
[link](https://github.com/megvii-research/NAFNet/blob/main/docs/SIDD.md)

## Rain13k[本次用到的数据集]
+ 链接
[train](https://drive.google.com/file/d/14BidJeG4nSNuFNFDf99K-7eErCq4i47t/view?usp=sharing)
[test](https://drive.google.com/file/d/1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8/view?usp=sharing)
+ 使用代码下载
[python](https://github.com/swz30/Restormer/blob/main/Deraining/download_data.py)


For training and testing, your directory structure should look like this
 `Datasets` <br/>
 `├──train`  <br/>
     `└──Rain13K`   <br/>
          `├──input`   <br/>
          `└──target`   <br/>
 `└──test`  <br/>
     `├──Test100`   <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `├──Rain100H`  <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `├──Rain100L`  <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `├──Test1200`  <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `└──Test2800`<br/>
          `├──input`   <br/>
          `└──target` 
# 单卡训练
```
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=2 python basicsr/train.py -opt options/train/Rain13k/Wavelet-width32.yml
```
# 多卡卡训练
```
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=1,2,3 python basicsr/train.py -opt options/train/Rain13k/Wavelet-width32.yml
```

# 清理模型
如果存储模型过多，请跑`bash clean.sh`
