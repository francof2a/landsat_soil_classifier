import os
import shutil
import argparse

MODEL_NAME = 'ANN50x50'
IP_SRC_PATH = '../fpga/hls_'+MODEL_NAME+'/fpga_'+MODEL_NAME+'_prj/solution1/impl/ip/'
IP_DST_PATH = '../ip/'

def ip_checkout(name, src_path=IP_SRC_PATH, dst_path=IP_DST_PATH):
    ip_folder = name+'_ip'
    try:
        print('exporting IP of {} from {} to {}'.format(name, src_path, dst_path+ip_folder))
        os.system('mkdir {dst}'.format(dst=dst_path+ip_folder))
        os.system('cp -r {src}* {dst}'.format(src=src_path, dst=dst_path+ip_folder))
        print('done!')
    except:
        print('IP checkout failed!')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FPGA IP checkout')
    parser.add_argument('-n', '--name', metavar='model_name', type=str, required=True, help='name of the model to do (FPGA) IP checkout')

    args = parser.parse_args()

    ip_checkout(args.name)