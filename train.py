from GAN_models.py import GAN, VAEGAN
import argparse
import numpy as np

# =============================== Load Dataset ============================= #

# Depending on what the data looks like, length/width/height should be adjusted
# Assuming all augmentation and pre-processing steps are done
# Training and testing splits remain the same

# length =
# width =
# height ='

# Normalize the data (not sure if necessary?)
# Split into training, validation, and test sets

# =============================== Parsing Cmd Arguments ============================= #
parser = argparse.ArgumentParser(description='Training 3DGAN:')

parser.add_argument('--model', type=str, help='3D_GAN or 3D_VAE_GAN')
parser.add_argument('--data', type=np.ndarray, help='training dataset')
parser.add_argument('--epochs', type=int, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, help='batch size for training')
parser.add_argument('--save_int', type=int, help='interval to save data at')
parser.add_argument('--out_file', type=str, help='file to save model')

args = parser.parse_args()
# tmp_folder = './tmp/'

# TODO: Save the model after training here?
def train_model():
    if (args.model == '3D_GAN'):
        basic_model = GAN()
        basic_model.train(args.data, args.epochs, args.batch_size, args.out_file)
    elif (args.model == '3D_VAE_GAN'):
        pass

def main():
    assert (args.model == '3D_GAN' or args.model == '3D_VAE_GAN'), 'Invalid model'
    # TODO: save model

if __name__ == '__main__':
    main()

