import torch, os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch.nn.functional as F
from PIL import Image
from models.baseline import BiRefNet
from config import Config
from torchvision.transforms.functional import normalize
import numpy as np
# import folder_paths

config = Config()

device = "cuda" if torch.cuda.is_available() else "cpu"
# folder_paths.folder_names_and_paths["BiRefNet"] = ([os.path.join(folder_paths.models_dir, "BiRefNet")], folder_paths.supported_pt_extensions)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

class MyBiRefNet:
    def __init__(self):
        self.net = BiRefNet()
        self.model_path = os.path.join('birefnet_model','BiRefNet-DIS_ep580.pth')
        self.state_dict = torch.load(self.model_path, map_location=device)
        self.unwanted_prefix = '_orig_mod.'
        for k, v in list(self.state_dict.items()):
            if k.startswith(self.unwanted_prefix):
                self.state_dict[k[len(self.unwanted_prefix):]] = self.state_dict.pop(k)
        self.net.load_state_dict(self.state_dict)
        self.net.to(device)
        self.net.eval() 
    

    def process(self, input_path):
        processed_images = []
        processed_masks = []
        output_dir = os.path.join('./output2', os.path.basename(input_path))
        os.makedirs(output_dir, exist_ok=True)

        for root, _, files in os.walk(input_path):
            for i, file in enumerate(sorted(files), start=1):
                if file.endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    orig_image = Image.open(image_path).convert('RGB')
                    w, h = orig_image.size
                    image = resize_image(orig_image)
                    im_np = np.array(image)
                    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
                    im_tensor = torch.unsqueeze(im_tensor, 0)
                    im_tensor = torch.divide(im_tensor, 255.0)
                    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
                    if torch.cuda.is_available():
                        im_tensor = im_tensor.cuda()

                    result = self.net(im_tensor)[-1].sigmoid()
                    result = torch.squeeze(F.interpolate(result, size=(h, w), mode='bilinear'), 0)
                    ma = torch.max(result)
                    mi = torch.min(result)
                    result = (result - mi) / (ma - mi)
                    im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
                    pil_im = Image.fromarray(np.squeeze(im_array))
                    new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
                    new_im.paste(orig_image, mask=pil_im)
                    new_im.save(os.path.join(output_dir, f"{i}.png"), format='PNG')
                    
