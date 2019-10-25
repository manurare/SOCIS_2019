import torchvision

from PIL import Image

from random import shuffle


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder_shuffle(torchvision.datasets.DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, train_folder, test_folder = None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        if test_folder is not None:
            all_samples = []
            super(ImageFolder_shuffle, self).__init__(train_folder, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                      transform=transform,
                                                      target_transform=target_transform)
            train_samples = self.samples
            super(ImageFolder_shuffle, self).__init__(test_folder, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                      transform=transform,
                                                      target_transform=target_transform)
            all_samples = train_samples + self.samples
            self.samples.clear()
            self.samples = all_samples
            shuffle(self.samples)
            self.imgs = self.samples
        else:
            super(ImageFolder_shuffle, self).__init__(train_folder, loader,
                                                      IMG_EXTENSIONS if is_valid_file is None else None,
                                                      transform=transform,
                                                      target_transform=target_transform)
            shuffle(self.samples)
            self.imgs = self.samples


