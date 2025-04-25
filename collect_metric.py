
import os, sys
import json
from argparse import ArgumentParser, Namespace

def get_folder_list(dataset):
    if dataset == "dycheck":
        folder_list = ["apple", "block", "spin", "paper-windmill", "space-out", "teddy", "wheel"]       
    elif dataset == "dynerf":
        folder_list = ["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"]
    elif dataset == "nvidia":
        folder_list = ["Balloon1", "Balloon2", "Jumping", "dynamicFace","Playground", "Skating", "Truck", "Umbrella"]
    elif dataset == "hypernerf":
        folder_list = ["aleks-teapot", "chickchicken", "cut-lemon1", "hand1", "slice-banana", "torchocolate", 
                       "americano", "cross-hands1", "espresso", "keyboard", "oven-mitts", "split-cookie", "tamping", 
                       "3dprinter", "broom", "chicken", "peel-banana"]
    elif dataset == "dnerf":
        folder_list = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex"]
    elif dataset == "panoptic_sports":
        folder_list = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    return folder_list


def collect_metric(folder_list, output_path):

    psnr_results = {}
    ssim_results = {}
    msssim_results = {}
    lpips_vgg_results = {}
    lpips_alex_results = {}    
    total_results = {}

    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            results = json.load(f)

        result_key = list(results.keys())[0]
                
        psnr_results[folder] = results[result_key]['PSNR']
        ssim_results[folder] = results[result_key]['SSIM']
        msssim_results[folder] = results[result_key]['MS-SSIM']
        lpips_vgg_results[folder] = results[result_key]['LPIPS-vgg']
        lpips_alex_results[folder] = results[result_key]['LPIPS-alex']
        
        total_results[folder] = results[result_key]

    output_folder_name = output_path.split("/")[-1]
    
    with open(os.path.join(output_path, output_folder_name+ "_total_results.json"), 'w') as f:
        json.dump(total_results, f)

    with open(os.path.join(output_path, output_folder_name+ "_total_results.txt"), 'w') as f:
        for key, value in total_results.items():
            f.write(f"{key} : {value}\n")

    # psnr
    with open(os.path.join(output_path, output_folder_name+ "_psnr_results.txt"), 'w') as f:
        for key, value in psnr_results.items():
            f.write(f"{key} : {value}\n")
            
    # ssim
    with open(os.path.join(output_path, output_folder_name+ "_ssim_results.txt"), 'w') as f:
        for key, value in ssim_results.items():
            f.write(f"{key} : {value}\n")
            
    # msssim
    with open(os.path.join(output_path, output_folder_name+ "_msssim_results.txt"), 'w') as f:
        for key, value in msssim_results.items():
            f.write(f"{key} : {value}\n")
            
    # lpips_vgg
    with open(os.path.join(output_path, output_folder_name+ "_lpips_vgg_results.txt"), 'w') as f:
        for key, value in lpips_vgg_results.items():
            f.write(f"{key} : {value}\n")
            
    # lpips_alex
    with open(os.path.join(output_path, output_folder_name+ "_lpips_alex_results.txt"), 'w') as f:
        for key, value in lpips_alex_results.items():
            f.write(f"{key} : {value}\n")
   
def collect_psnr_ssim_lpips_memory(folder_list, output_path):
    
    psnr_results = {}
    ssim_results = {}
    lpips_results= {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]

    with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue
            
        with open(json_path) as f:
            results = json.load(f)

        result_key = list(results.keys())[0]      
        
        psnr_results[folder] = results[result_key]['PSNR']
        ssim_results[folder] = results[result_key]['SSIM']
        lpips_results[folder] = results[result_key]['LPIPS-vgg']

        model_folder_list = os.listdir(model_path)
        model_folder_list.sort()
        model_folder = model_folder_list[-1]
        
        total_path = os.path.join(model_path, model_folder)
        
        total_size = sum(os.path.getsize(os.path.join(total_path, f)) for f in os.listdir(total_path)) / (1000*1000)
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB")

        with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB\n")
            
    

def collect_psnr_msssim_lpips_memory(folder_list, output_path):
    
    psnr_results = {}
    ssim_results = {}
    lpips_results= {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]

    with open(os.path.join(output_path, output_folder_name+ "_psnr_msssim_lpips_memory.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_psnr_msssim_lpips_memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue
            
        with open(json_path) as f:
            results = json.load(f)
        result_key = list(results.keys())[0]      
        
        psnr_results[folder] = results[result_key]['PSNR']
        ssim_results[folder] = results[result_key]['MS-SSIM']
        lpips_results[folder] = results[result_key]['LPIPS-vgg']

        model_folder_list = os.listdir(model_path)

        model_folder_list.sort()

        model_folder = model_folder_list[-1]

        total_path = os.path.join(model_path, model_folder)

        total_size = sum(os.path.getsize(os.path.join(total_path, f)) for f in os.listdir(total_path)) / (1000*1000)
        # print(f"{folder} : {total_size} MB")
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['MS-SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB")

        with open(os.path.join(output_path, output_folder_name+ "_psnr_msssim_lpips_memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['MS-SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB\n")



def collect_memory(folder_list, output_path):
    
    total_memory = {}
    
    for folder in folder_list:
        model_path = os.path.join(output_path, folder, "point_cloud")

        if not os.path.exists(model_path):
            continue

        model_folder_list = os.listdir(model_path)
        model_folder_list.sort()
        model_folder = model_folder_list[-1]

        total_path = os.path.join(model_path, model_folder)

        total_size = sum(os.path.getsize(os.path.join(total_path, f)) for f in os.listdir(total_path)) / (1000*1000)

        deform_path = os.path.join(output_path, folder, "deform")
        if os.path.exists(deform_path):
            deform_folder_list = os.listdir(deform_path)
            deform_folder_list.sort()
            deform_folder = deform_folder_list[-1]
            deform_total_path = os.path.join(deform_path, deform_folder)
            deform_total_size = sum(os.path.getsize(os.path.join(deform_total_path, f)) for f in os.listdir(deform_total_path)) / (1000*1000)
            total_size += deform_total_size
        
        
        total_memory[folder] = total_size
    
    output_folder_name = output_path.split("/")[-1]
        
    with open(os.path.join(output_path, output_folder_name+ "_ total_memory.txt"), 'w') as f:
        for key, value in total_memory.items():
            f.write(f"{key} : {value}\n")

def merge_psnr_and_memory(folder_list, output_path):
    
    psnr_results = {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]

    with open(os.path.join(output_path, output_folder_name+ "_ psnr_and_memory.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_ psnr_and_memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue

        with open(json_path) as f:
            results = json.load(f)

        result_key = list(results.keys())[0]      
        psnr_results[folder] = results[result_key]['PSNR']

        model_folder_list = os.listdir(model_path)
        model_folder_list.sort()
        model_folder = model_folder_list[-1]
        
        total_path = os.path.join(model_path, model_folder)
        
        total_size = sum(os.path.getsize(os.path.join(total_path, f)) for f in os.listdir(total_path)) / (1000*1000)

        deform_path = os.path.join(output_path, folder, "deform")
        if os.path.exists(deform_path):
            deform_folder_list = os.listdir(deform_path)
            deform_folder_list.sort()
            deform_folder = deform_folder_list[-1]
            deform_total_path = os.path.join(deform_path, deform_folder)
            deform_total_size = sum(os.path.getsize(os.path.join(deform_total_path, f)) for f in os.listdir(deform_total_path)) / (1000*1000)
            total_size += deform_total_size
        
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['PSNR']} {total_size} MB")


        with open(os.path.join(output_path, output_folder_name+ "_ psnr_and_memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['PSNR']} {total_size} MB\n")
        

def merge_masked_results(folder_list, output_path):
    
    psnr_results = {}
    ssim_results = {}
    lpips_results= {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]
    
    with open(os.path.join(output_path, output_folder_name+ "_mPSNR_and_Memory.txt"), 'w') as f:
        f.write("")
    
    with open(os.path.join(output_path, output_folder_name+ "_mPSNR_mSSIM_mLPIPS_and_Memory.txt"), 'w') as f:
        f.write("")
        
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results_masked.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_mPSNR_and_Memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            with open(os.path.join(output_path, output_folder_name+ "_mPSNR_mSSIM_mLPIPS_and_Memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")               
            continue

        with open(json_path) as f:
            results = json.load(f)

        result_key = list(results.keys())[0]      

        psnr_results[folder] = results[result_key]['mPSNR']
        ssim_results[folder] = results[result_key]['mSSIM']
        lpips_results[folder] = results[result_key]['mLPIPS']

        model_folder_list = os.listdir(model_path)
        model_folder_list.sort()
        model_folder = model_folder_list[-1]

        total_path = os.path.join(model_path, model_folder)
        
        total_size = sum(os.path.getsize(os.path.join(total_path, f)) for f in os.listdir(total_path)) / (1000*1000)
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['mPSNR']} {total_size} MB")

        with open(os.path.join(output_path, output_folder_name+ "_mPSNR_and_Memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['mPSNR']} {total_size} MB\n")    
        
        with open(os.path.join(output_path, output_folder_name+ "_mPSNR_mSSIM_mLPIPS_and_Memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['mPSNR']} {results[result_key]['mSSIM']} {results[result_key]['mLPIPS']} {total_size} MB\n")    


def merge_masked_lpips_vgg(folder_list, output_path):
    
    lpips_results= {}
    
    output_folder_name = output_path.split("/")[-1]

    with open(os.path.join(output_path, output_folder_name+ "_mLPIPS-vgg.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results_masked_lpips.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_mLPIPS-vgg.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue
        with open(json_path) as f:
            results = json.load(f)

        result_key = list(results.keys())[0]      

        lpips_results[folder] = results[result_key]['mLPIPS']
       
        print(f"{folder} : {results[result_key]['mLPIPS']}")
        with open(os.path.join(output_path, output_folder_name+ "_mLPIPS-vgg.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['mLPIPS']}\n")
        
   
        
        
if __name__ == "__main__":
        
    parser = ArgumentParser(description="collection parameters")
    
    parser.add_argument('--output_path', type=str, default="./output/dynerf_anchor")
    parser.add_argument('--dataset', type=str, default="dycheck")
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--lpips_only', type=int, default=0)
    parser.add_argument('--all', type=int, default=0)
    

    args = parser.parse_args(sys.argv[1:])

    folder_list = get_folder_list(args.dataset)

    #collect_metric(folder_list, args.output_path)

    #collect_memory(folder_list, args.output_path)
    
    collect_psnr_ssim_lpips_memory(folder_list, args.output_path)
    
    collect_psnr_msssim_lpips_memory(folder_list, args.output_path)


    if args.mask:
	    if args.lpips_only:
	        merge_masked_lpips_vgg(folder_list, args.output_path)
	    else:        
	        merge_masked_results(folder_list, args.output_path)
    else:
	    #merge_psnr_and_memory(folder_list, args.output_path)
        collect_psnr_ssim_lpips_memory(folder_list, args.output_path)
 		
    
    




