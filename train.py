import torch
import numpy as np
import os

from model import Net
from utils import get_dataloader, get_image_shape, get_config, get_char_dict, red_print, blue_print, green_print, \
    yellow_print, purple_print, convert_test_output


config = './configs/global.yml'

global_config, *_ = get_config(config)
device = 'cuda' if torch.cuda.is_available() and global_config['use_gpu'] else 'cpu'
epoch_num, save_model_dir, save_epoch_step, learning_rate = \
    global_config['epoch_num'], \
    global_config['save_model_dir'], \
    global_config['save_epoch_step'], \
    global_config['learning_rate']

train_dataloader, test_dataloader = get_dataloader(config)
blue_print(f'Train dataset size: {len(train_dataloader)}')
blue_print(f'Test dataset size: {len(test_dataloader)}')

image_shape = get_image_shape(config)
blue_print(f'Image shape: {image_shape}')

char_dict = get_char_dict(global_config['character_dict_path'])
blue_print(f'Character dict length: {len(char_dict)}')

model = Net(image_shape, len(char_dict)).to(device)

criterion = torch.nn.CTCLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

red_print(f'Start training with {"gpu" if device == "cuda" else "cpu"}')
for epoch in range(1, epoch_num + 1):
    model.train()
    total_loss = []
    for batch_idx, (image, value, length) in enumerate(train_dataloader):
        image, value = image.to(device), value.to(device)
        optimizer.zero_grad()
        output = model(image)
        input_lengths = torch.IntTensor([output.shape[0]] * output.shape[1])
        loss = criterion(output, value, input_lengths, length)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    green_print(f'epoch: {epoch}, total_loss: {np.mean(total_loss)}')

    model.eval()
    acc_num, total_num = 0, 0
    with torch.no_grad():
        for batch_idx, (image, value, length) in enumerate(test_dataloader):
            image, value = image.to(device), value.to(device)
            output = model(image)
            output = output.permute(1, 0, 2)
            for i in range(output.shape[0]):
                predict = convert_test_output(output[i], char_dict)
                label = ''.join([char_dict[num] for num in value[i].tolist() if num != 0])
                if predict == label:
                    acc_num += 1
                total_num += 1

    yellow_print(f'epoch: {epoch}, accuracy: {acc_num / total_num:.5f}')

    if epoch % save_epoch_step == 0 and epoch != 0:
        torch.save(model, os.path.join(save_model_dir, f'epoch_{epoch}.pth'))
        purple_print(f'model had saved in epoch_{epoch}.pth.')
