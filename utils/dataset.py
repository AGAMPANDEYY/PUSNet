import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import torch
import os
from PIL import Image
import config as c

class dataset_(Dataset):
    def __init__(self, img_dir, transform, sigma):
        self.img_dir = img_dir
        self.img_filenames = list(sorted(os.listdir(img_dir)))
        self.transform = transform
        self.sigma = sigma
        self.totensor = T.ToTensor()
    
    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, index):
        img_paths = os.path.join(self.img_dir, self.img_filenames[index])
        img = Image.open(img_paths).convert("RGB")
        img = self.transform(img)
        if self.sigma != None:
            noised_img = img + torch.randn(img.shape).mul_(self.sigma/255)
            return img, noised_img
        return img

class PairedImageFolder(Dataset):
    def __init__(self, cover_dir, secret_dir,sigma, transform=None):
        self.cover_dir = cover_dir
        self.secret_dir = secret_dir
        self.transform = transform
        self.sigma=sigma
        
        self.cover_images = sorted([
            os.path.join(cover_dir, f)
            for f in os.listdir(cover_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.secret_images = sorted([
            os.path.join(secret_dir, f)
            for f in os.listdir(secret_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        assert len(self.cover_images) == len(self.secret_images), \
            "Mismatch between number of cover and secret images"

    def __len__(self):
        return len(self.cover_images)

    def __getitem__(self, idx):
        cover = Image.open(self.cover_images[idx]).convert("RGB")
        secret = Image.open(self.secret_images[idx]).convert("RGB")

        if self.transform:
            cover = self.transform(cover)
            secret = self.transform(secret)
        if self.sigma != None:
            noised_img_cover = cover + torch.randn(cover.shape).mul_(self.sigma/255)
            noised_img_secret= secret + torch.randn(secret.shape).mul_(self.sigma/255)
            return cover, noised_img_cover, secret, noised_img_secret

        return cover, secret


transform_train = T.Compose([
    T.RandomCrop(c.crop_size_train),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    # T.RandomCrop(c.crop_size_train),
    T.ToTensor()
])

transform_val = T.Compose([
    # T.CenterCrop(c.crop_size_train),
    T.Resize([c.resize_size_test, c.resize_size_test]),
    T.ToTensor(),
])

test_data_dir_secret="/kaggle/input/steganaylsis/steganalaysis/secret"
test_data_dir_cover="/kaggle/input/steganaylsis/steganalaysis/cover"

def load_dataset(train_data_dir, test_data_dir, batchsize_train, batchsize_test, sigma=None):

    test_loader = DataLoader(
        PairedImageFolder(test_data_dir_cover,test_data_dir_secret ,sigma,transform_val),
        batch_size=batchsize_test,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=True
    )

    return  test_loader


# transform_train = A.Compose(
#     [
#         A.RandomCrop(128, 128),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         ToTensorV2(),
#     ]
# )

# transform_val = A.Compose([
#     A.CenterCrop(256, 256),
#     ToTensorV2(),
# ])


# class dataset_(Dataset):
#     def __init__(self, img_dir, sigma, transform):
#         self.img_dir = img_dir
#         self.img_filenames = list(sorted(os.listdir(img_dir)))
#         self.sigma = sigma
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_filenames)

#     def __getitem__(self, idx):
#         img_filename = self.img_filenames[idx]
#         img = cv2.imread(os.path.join(self.img_dir, img_filename))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img/255)
        
#         if self.transform:
#             img = self.transform(image=img)["image"]
            
#         noised_img = img + torch.randn(img.shape).mul_(self.sigma/255)

#         return img, noised_img


# def load_dataset(train_data_dir, test_data_dir, batch_size, sigma=None):

#     train_loader = DataLoader(
#         dataset_(train_data_dir, sigma, transform_train),
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=8,
#         drop_last=True
#     )

#     test_loader = DataLoader(
#         dataset_(test_data_dir, sigma, transform_val),
#         batch_size=2,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=1,
#         drop_last=True
#     )

#     return train_loader, test_loader


    
