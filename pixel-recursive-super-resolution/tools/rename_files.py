import os
from PIL import Image
path = '/home/manuelrey/ESA/Dataset/Step2-SuperresolutionWhale/Whale'
path_to_save = '/home/manuelrey/ESA/Dataset/Step2-SuperresolutionWhale/converted'
# path = '/home/manuelrey/ESA/Dataset/Step1-PresenceWhale/Whale'
# path_to_save = '/home/manuelrey/ESA/Dataset/Step1-PresenceWhale/whale_test'
files = os.listdir(path)

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
for index, file in enumerate(files):
    if file[0] != '.':
        im = Image.open(os.path.join(path, file))
        im = im.convert('RGB')
        im.save(os.path.join(path_to_save, '%04d.png' % index))