'''DNN Feature extraction script'''

# %%%
from glob import glob
from tqdm import tqdm
import os
import argparse
from PIL import Image
import numpy as np
import torch
from scipy.io import savemat
from bdpy.dl.torch import FeatureExtractor
from torchvision import models, transforms
import yaml
# %%%
# Custum transform class to convert RGB to BGR 
class ConvertRGBtoBGR:
    def __call__(self, image):
        return image[[2, 1, 0], :, :]  
    
# Custum transform class to convert Image format to tensor while keeping pixel ranges
class ToTensorWithoutScaling:
    def __call__(self, image):
        image = np.array(image).astype(np.float32)  # PIL ImageをNumPy配列に変換
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # NumPy配列をテンソルに変換し、次元を並べ替える
        return tensor
    
class Convert32to16:
    def __call__(self, image):
        return image.type(torch.float16)

# %%%
def load_model(network):
    
    if network == "VGG19_ILSVRC_19_layers":
        from bdpy.dl.torch.models import layer_map, model_factory
        model = model_factory('vgg19')
        encoder_param_file = './data/models_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'
        model.load_state_dict(torch.load(encoder_param_file))
        model.eval()
        layer_mapping = layer_map('vgg19')
        # change RGB-> BGR and subtract mean via lambda function
        mean_image = [104., 117., 123.] #BGR!
        #mean_image = [103.939, 116.779, 123.68] #BGR!
        preprocess = transforms.Compose([
                    transforms.Resize(224, interpolation=Image.BICUBIC),
                    #transforms.CenterCrop(224),
                    #transforms.ToTensor(),
                    ToTensorWithoutScaling(), # カスタム変換を追加
                    ConvertRGBtoBGR(),  # カスタム変換を追加
                    transforms.Normalize(mean=mean_image, std=[1.,1.,1.]),
                ])
        
    elif network == "vgg19_torchvision":
        model = models.vgg19(pretrained=True)
        model.eval()
        
        # 画像の前処理
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) 
    else:
        raise ValueError("Network not supported. Please choose from: VGG19_ILSVRC_19_layers"
                        +"or define by yourself. The output should be model and preprocessing functions from RGB image to model input."
                         )
    
    return model, preprocess

def load_config(config_path):
    '''Load configuration file.'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_features(config, device='cuda'):
    '''Extract features based on the configuration.'''
    print("Extracting features using network:", config['network'])
    
    image_dir = config['image path']
    image_ext = config["image_ext"]
    image_files = glob(os.path.join(image_dir, '*.' + image_ext))
    
    model, preprocess = load_model(config['network'])
    layers = config["features"]
    feature_extractor = FeatureExtractor(model, layers, device=device, detach=True)
    
    output_dir = os.path.join(config["output base dir"], "pytorch",config['network'])
    
    for image_file in tqdm(image_files):
        img = Image.open(image_file).convert('RGB')
        
        x = preprocess(img).unsqueeze(0).to(device)
        
        # Extract features
        features = feature_extractor.run(x)

        # Save features
        for layer in features.keys():
            f = features[layer]

            output_file = os.path.join(
                output_dir,
                layer,
                os.path.splitext(os.path.basename(image_file))[0] + '.mat'
            )

            if os.path.exists(output_file):
                continue

            os.makedirs(os.path.join(output_dir, layer), exist_ok=True)

            savemat(output_file, {'feat': f})
        #print('Saved {}'.format(output_file))

    print('All done')

        
    
# %%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Feature extraction')
    parser.add_argument('config', help='Configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    extract_features(config)
# %%