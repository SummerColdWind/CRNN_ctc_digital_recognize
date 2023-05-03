import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from .get_char_dict import get_char_dict
from .get_config import get_config


class DigitalDataset(Dataset):
    def __init__(
            self,
            data_path,
            label_path,
            char_dict,
            seq_max_length,
            transform=None
    ):
        super().__init__()
        self.root_path = data_path
        self.label_path = label_path
        self.char_dict = char_dict
        self.seq_max_length = seq_max_length
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.data = []
        with open(label_path, 'r') as file:
            for line in file.readlines():
                image, value = line.strip().split('\t')
                self.data.append((os.path.join(data_path, image), value))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image, value = self.data[item]  # For example: torch.Size([1, 21, 44])
        image = self.transform(Image.open(image).convert('RGB'))
        value = torch.as_tensor(
            [self.char_dict.index(char) for char in f'{value: ^{self.seq_max_length}}'],
            dtype=torch.int64
        )
        return image, value, len(value)



def get_dataloader(config_path):
    global_config, train_config, test_config = get_config(config_path)
    char_dict = get_char_dict(global_config['character_dict_path'])
    seq_max_length = global_config['max_text_length']
    train_dataset = DigitalDataset(
        data_path=train_config['data_dir'],
        label_path=train_config['label_file_dir'],
        char_dict=char_dict,
        seq_max_length=seq_max_length
    )
    test_dataset = DigitalDataset(
        data_path=test_config['data_dir'],
        label_path=test_config['label_file_dir'],
        char_dict=char_dict,
        seq_max_length=seq_max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=train_config['shuffle']
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_config['batch_size'],
        shuffle=test_config['shuffle']
    )
    return train_dataloader, test_dataloader


