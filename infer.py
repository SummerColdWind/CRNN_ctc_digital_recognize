import torch
from torchvision import transforms
from PIL import Image

from utils import get_config, get_char_dict, convert_infer_output

config = './configs/global.yml'
model_path = './output/best.pth'
image_path = './example.png'

global_config, *_ = get_config(config)
device = 'cuda' if torch.cuda.is_available() and global_config['use_gpu'] else 'cpu'
char_dict = get_char_dict(global_config['character_dict_path'])

image = Image.open(image_path).convert('RGB')
image = transforms.ToTensor()(image)
image = torch.reshape(image, (1, ) + tuple(image.shape))
image = image.to(device)

model = torch.load(model_path)
model.eval()

output = model(image)
predict = convert_infer_output(output, char_dict)

print(f'Predict: {predict}')


