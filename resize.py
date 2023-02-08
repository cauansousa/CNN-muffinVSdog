from PIL import Image
import os

path = "C:\\Users\\Cauan\\Documents\\GitHub\\Muffin-Chihuahua-CNN\\foto\\"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((256,256), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

resize()