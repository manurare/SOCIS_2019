from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

########## OLD
# def calculate_valid_crop_size(crop_size, upscale_factor):
#     return crop_size - (crop_size % upscale_factor)
#
#
# def train_hr_transform(crop_size):
#     return Compose([
#         RandomCrop(crop_size),
#         ToTensor(),
#     ])
#
#
# def train_lr_transform(crop_size, upscale_factor):
#     return Compose([
#         ToPILImage(),
#         Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
#         ToTensor()
#     ])

# class TrainDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, crop_size, upscale_factor):
#         super(TrainDatasetFromFolder, self).__init__()
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
#         self.hr_transform = train_hr_transform(crop_size)
#         self.lr_transform = train_lr_transform(crop_size, upscale_factor)
#
#     def __getitem__(self, index):
#         hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
#         lr_image = self.lr_transform(hr_image)
#         return lr_image, hr_image
#
#     def __len__(self):
#         return len(self.image_filenames)
#
#
# class ValDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(ValDatasetFromFolder, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
#
#     def __getitem__(self, index):
#         hr_image = Image.open(self.image_filenames[index])
#         w, h = hr_image.size
#         crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
#         lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
#         hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
#         hr_image = CenterCrop(crop_size)(hr_image)
#         lr_image = lr_scale(hr_image)
#         hr_restore_img = hr_scale(lr_image)
#         return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
#
#     def __len__(self):
#         return len(self.image_filenames)
#
#
# class TestDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(TestDatasetFromFolder, self).__init__()
#         self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
#         self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
#         self.upscale_factor = upscale_factor
#         self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
#         self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]
#
#     def __getitem__(self, index):
#         image_name = self.lr_filenames[index].split('/')[-1]
#         lr_image = Image.open(self.lr_filenames[index])
#         w, h = lr_image.size
#         hr_image = Image.open(self.hr_filenames[index])
#         hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
#         hr_restore_img = hr_scale(lr_image)
#         return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
#
#     def __len__(self):
#         return len(self.lr_filenames)


def train_hr_transform(hr_size):
    return Compose([
        Resize(size=(hr_size, hr_size),interpolation=Image.BICUBIC),
        ToTensor(),
    ])


def train_lr_transform(hr_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(size=(hr_size // upscale_factor, hr_size // upscale_factor), interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder_dataAug(Dataset):
    def __init__(self, dataset_dir_aug, dataset_dir, hr_size, upscale_factor):
        super(TrainDatasetFromFolder_dataAug, self).__init__()
        self.image_filenames = []
        for filename in listdir(dataset_dir):
            if is_image_file(filename):
                self.image_filenames.append(join(dataset_dir, filename))
        for filename in listdir(dataset_dir_aug):
            if is_image_file(filename):
                self.image_filenames.append(join(dataset_dir_aug, filename))
        self.hr_transform = train_hr_transform(hr_size)
        self.lr_transform = train_lr_transform(hr_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, hr_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.hr_transform = train_hr_transform(hr_size)
        self.lr_transform = train_lr_transform(hr_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, hr_size, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.hr_size = hr_size

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        lr_scale = Resize(size=(self.hr_size // self.upscale_factor, self.hr_size // self.upscale_factor)
                          , interpolation=Image.BICUBIC)
        hr_scale = Resize(size=(self.hr_size,self.hr_size), interpolation=Image.BICUBIC)
        hr_image = hr_scale(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


# class TestDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, hr_size, upscale_factor):
#         super(TestDatasetFromFolder, self).__init__()
#         # self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
#         # self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
#         self.path = dataset_dir
#         self.hr_size = hr_size
#         self.upscale_factor = upscale_factor
#         self.hr_transform = train_hr_transform(hr_size)
#         self.lr_transform = train_lr_transform(hr_size, upscale_factor)
#         self.filenames = [join(self.path, x) for x in listdir(self.path) if is_image_file(x)]
#
#     def __getitem__(self, index):
#         image_name = self.filenames[index].split('/')[-1]
#         hr_image = self.hr_transform(Image.open(self.filenames[index]))
#         lr_image = self.lr_transform(hr_image)
#         hr_scale = Resize(size=(self.hr_size,self.hr_size), interpolation=Image.BICUBIC)
#         lr_resized2hr = hr_scale(ToPILImage()(lr_image))
#         return image_name, lr_image, ToTensor()(lr_resized2hr), hr_image
#
#     def __len__(self):
#         return len(self.filenames)

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.filenames[index].split('/')[-1]
        lr_image = Image.open(self.filenames[index])
        w, h = lr_image.size
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        lr_in_hr_scale = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(lr_in_hr_scale)

    def __len__(self):
        return len(self.filenames)
