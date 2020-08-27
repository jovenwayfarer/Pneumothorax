from train import Pneumothorax
import torch
import glob


ckpt_models = glob.glob('weights/*.ckpt')
for i, checkpoint_path in enumerate(ckpt_models):
    
    model = Pneumothorax.load_from_checkpoint(checkpoint_path)
    model.eval()
    state_dict = model.state_dict()
    torch.save(state_dict, checkpoint_path[:-4]+'pth')

