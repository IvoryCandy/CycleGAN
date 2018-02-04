from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import os
import random


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, sub_folder='train', transform=None, resize_scale=None, crop_size=None, flip=False):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, sub_folder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.flip = flip

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn).convert('RGB')

        # pre-processing
        if self.resize_scale:
            img = img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))

        if self.flip:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_filenames)


def train_dataloader(input_size, batch_size, dataset):
    data_dir = './datasets/' + dataset + '/'
    transform = transforms.Compose([transforms.Resize(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # Test data
    test_data_A = DatasetFromFolder(data_dir, sub_folder='trainA', transform=transform)
    test_data_loader_A = data.DataLoader(dataset=test_data_A, batch_size=batch_size, shuffle=False)
    test_data_B = DatasetFromFolder(data_dir, sub_folder='trainB', transform=transform)
    test_data_loader_B = data.DataLoader(dataset=test_data_B, batch_size=batch_size, shuffle=False)

    return test_data_loader_A, test_data_loader_B


def test_dataloader(input_size, batch_size, dataset):
    data_dir = './datasets/' + dataset + '/'
    transform = transforms.Compose([transforms.Resize(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # Test data
    test_data_A = DatasetFromFolder(data_dir, sub_folder='testA', transform=transform)
    test_data_loader_A = data.DataLoader(dataset=test_data_A, batch_size=batch_size, shuffle=False)
    test_data_B = DatasetFromFolder(data_dir, sub_folder='testB', transform=transform)
    test_data_loader_B = data.DataLoader(dataset=test_data_B, batch_size=batch_size, shuffle=False)

    return test_data_loader_A, test_data_loader_B
