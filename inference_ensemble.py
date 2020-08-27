from torch.utils.data import DataLoader, Dataset
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch
import pandas as pd
import segmentation_models_pytorch as smp
from collections import OrderedDict
from tqdm import tqdm
import os
import numpy as np
import glob
from argparse import ArgumentParser



def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=24)
    return parser.parse_args()


class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.transform = albu.Compose(
            [
                albu.Normalize(mean=mean, std=std, p=1),
                albu.Resize(size, size),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples

def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle

def update_state_dict(state_dict):
    
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        new_state_dict['.'.join(key.split('.')[1:])] = state_dict[key]
    
    return new_state_dict




def main(args):  
    
    path_models = glob.glob('pth_weights/*.pth')

    sample_submission_path = 'labels/stage_2_sample_submission.csv'
    test_data_folder = 'size1024/test'

    size = 1024
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    best_threshold = 0.65
    min_size = 3500
    device = torch.device('cuda:{}'.format(args.gpu))
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(test_data_folder, df, size, mean, std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


    probs=[]
    model = smp.Unet("resnet34", encoder_weights=None, activation=None)
    for i, batch in enumerate(tqdm(testset)):
        
        for j in range(len(path_models)): #average over 2 folds

            state_dict = torch.load(path_models[j], map_location=lambda storage, loc: storage)
            state_dict = update_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model.to(device)
            
            if j==0:
                preds = torch.sigmoid(model(batch.to(device)))
            else:
                preds = preds+torch.sigmoid(model(batch.to(device)))
            
            model.cpu()
        
        preds=preds/len(path_models)
        preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
        for probability in preds:
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            probs.append(probability) 

    encoded_pixels = []

    for probability in probs:
        predict, num_predict = post_process(probability, best_threshold, min_size)
        if num_predict == 0:
            encoded_pixels.append('-1')
        else: 
            r = run_length_encode(predict)
            encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv('subs/5fold_ensemble.csv', columns=['ImageId', 'EncodedPixels'], index=False)


if __name__ == '__main__':
    args = get_args()
    main(args)
