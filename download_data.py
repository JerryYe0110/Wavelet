### Restormer: Efficient Transformer for High-Resolution Image Restoration
### Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
### https://arxiv.org/abs/2111.09881
#
### Download training and testing data for image deraining task
#import os
#import gdown
#import shutil
#
#import argparse
#
##parser = argparse.ArgumentParser()
##parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
##args = parser.parse_args()
#
#### Google drive IDs ######
#rain13k_train = '1P8nY17-upYmYOgPhFqIbet_zfSITM92o'   ## https://drive.google.com/file/d/14BidJeG4nSNuFNFDf99K-7eErCq4i47t/view?usp=sharing
#rain13k_test  = '1YnWS9s-unisIngTY_rPwlSVe0o3eBwQj'   ## https://drive.google.com/file/d/1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8/view?usp=sharing
#
#print('Rain13K Training Data!')
#gdown.download(id=rain13k_train, output='GT-RAIN.zip', quiet=False)
##os.system(f'gdrive download {rain13k_train} --path Datasets/')
#print('Extracting Rain13K data...')
#shutil.unpack_archive('GT-RAIN.zip', '.')
#os.remove('GT-RAIN.zip')
#
##for data in args.data.split('-'):
##    if data == 'train':
##        print('Rain13K Training Data!')
##        gdown.download(id=rain13k_train, output='Datasets/train.zip', quiet=False)
##        os.system(f'gdrive download {rain13k_train} --path Datasets/')
##        print('Extracting Rain13K data...')
##        shutil.unpack_archive('Datasets/train.zip', 'Datasets')
##        os.remove('Datasets/train.zip')
##
##    if data == 'test':
##        print('Download Deraining Testing Data')
##        gdown.download(id=rain13k_test, output='Datasets/test.zip', quiet=False)
##        os.system(f'gdrive download {rain13k_test} --path Datasets/')
##        print('Extracting test data...')
###        shutil.unpack_archive('Datasets/test.zip', 'Datasets')
##        os.remove('Datasets/test.zip')
#
#print('Download completed successfully!')

## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

## Download training and testing data for Image Denoising task


import os
import gdown
import shutil

### Google drive IDs ######
SIDD_train = '1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw'      ## https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing
SIDD_val   = '1gZx_K2vmiHalRNOb1aj93KuUQ2guOlLp'      ## https://drive.google.com/file/d/1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ/view?usp=sharing

print('SIDD Training Data!')
#os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
gdown.download(id=SIDD_train, output='Datasets/train.zip', quiet=False)
print('Extracting SIDD Data...')
shutil.unpack_archive('Datasets/train.zip', 'Datasets')
os.rename(os.path.join('Datasets', 'train'), os.path.join('Datasets', 'SIDD'))
os.remove('Datasets/train.zip')

print('SIDD Validation Data!')
gdown.download(id=SIDD_val, output='Datasets/val.zip', quiet=False)
print('Extracting SIDD Data...')
shutil.unpack_archive('Datasets/val.zip', 'Datasets')
os.remove('Datasets/val.zip')
