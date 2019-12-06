import os
import shutil
import argparse


MODEL_SRC_PATH = '../outputs/'
MODEL_DST_PATH = '../models/'

def model_checkout(name, src_path=MODEL_SRC_PATH, dst_path=MODEL_DST_PATH):
    model_json = name + '.json'
    model_h5 = name + '.h5'
    model_weights = name + '_weights.h5'

    try:
        print('exporting {} from {} to {}'.format(name, src_path, dst_path))
        print('\tmodel json: {}'.format(model_json))
        shutil.copyfile(src_path + model_json, dst_path + model_json)
        print('\tmodel h5: {}'.format(model_h5))
        shutil.copyfile(src_path + model_h5, dst_path + model_h5)
        print('\tmodel weights: {}'.format(model_weights))
        shutil.copyfile(src_path + model_weights, dst_path + model_weights)

        print('model checkout done!')
    except:
        print('Model checkout failed!')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model checkout')
    parser.add_argument('-n', '--name', metavar='model_name', type=str, required=True, help='name of the model to checkout')

    args = parser.parse_args()

    model_checkout(args.name)
