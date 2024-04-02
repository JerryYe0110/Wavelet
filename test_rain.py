import os
import yaml
import time
import warnings
from basicsr.test import main
import statistics
import torch
warnings.filterwarnings("ignore")

base_dir = 'experiments'
# dir_name = 'WaveletNet-Rain13k-width32'
yml_path = 'options/test/Rain13k/Wavelet-width32.yml'

test_datasets_base = "dataset/test/"
test_datasets_list = ["Test100", "Rain100H", "Rain100L", "Test1200", "Test2800"]

model_dict = {}
for dir_path in os.listdir(base_dir):
    if dir_path == "WaveletNet-Rain13k-width32_VanillaAttention_LocalConvNext_True":
        model_dir = os.path.join(base_dir, dir_path, "models")
        best_model_file = None
        best_avg_psnr = 0.
        best_test100 = ("", 0.)
        best_rain100H = ("", 0.)
        best_rain100L = ("", 0.)
        best_test1200 = ("", 0.)
        best_test2800 = ("", 0.)

        for model_fname in os.listdir(model_dir):
            
            model_fpath = os.path.join(model_dir, model_fname)

            with open(yml_path, 'r', encoding='utf-8') as f:
                opt = f.read()
                d = yaml.load(opt, Loader=yaml.FullLoader)
            g = dir_path.split("_")[1]
            l = dir_path.split("_")[2]
            s = dir_path.split("_")[3]
            d["network_g"]["G"] = g
            d["network_g"]["L"] = l
            d["network_g"]["S"] = bool(s)
            # assign new pretrained model path
            d["path"]["pretrain_network_g"] = model_fpath
            d["name"] = dir_path
            this_model_psnr = []
            
            for ds in test_datasets_list:
                ds_path = os.path.join(test_datasets_base, ds)
                ds_gt_path = ds_path+"/"+"target"
                ds_lq_path = ds_path+"/"+"input"
                # assign dataset path
                d["datasets"]["val"]["dataroot_gt"] = ds_gt_path
                d["datasets"]["val"]["dataroot_lq"] = ds_lq_path
                with open(yml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(d, f, default_flow_style=False)
                print(f">>>Testing Model<<<\n\t{model_fpath} on {ds}")
                psnr = main()
                print(f"Finish with psnr {psnr}")
                this_model_psnr.append(psnr)
                if ds == "Test100" and psnr>best_test100[1]:
                    best_test100 = (dir_path+"\t"+model_fname, psnr)
                elif ds == "Rain100H" and psnr>best_rain100H[1]:
                    best_rain100H = (dir_path+"\t"+model_fname, psnr)
                elif ds == "Rain100L" and psnr>best_rain100L[1]:
                    best_rain100L = (dir_path+"\t"+model_fname, psnr)
                elif ds == "Test1200" and psnr>best_test1200[1]:
                    best_test1200 = (dir_path+"\t"+model_fname, psnr)
                elif ds == "Test2800" and psnr>best_test2800[1]:
                    best_test2800 = (dir_path+"\t"+model_fname, psnr)
                    
                torch.cuda.empty_cache()
                # run python testing command
                # os.system("python basicsr/test.py -opt options/test/Rain13k/Wavelet-width32.yml")
                # time.sleep(2)
            psnr = statistics.mean(this_model_psnr)
            if psnr > best_avg_psnr:
                best_avg_psnr = psnr
                best_model_file = model_fname
        model_dict[str(dir_path)] = str(best_model_file)+"\t"+str(best_avg_psnr)
        print(f">>>>Here is the results for {dir_path}<<<<<")
        print(f"Test100: {best_test100}")
        print(f"Rain100H: {best_rain100H}")
        print(f"Rain100L: {best_rain100L}")
        print(f"Test1200: {best_test1200}")
        print(f"Test2800: {best_test2800}")
    else:
        pass

