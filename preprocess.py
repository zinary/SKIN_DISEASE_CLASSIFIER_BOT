import os
import cv2
from tqdm import tqdm
DIR = '/home/zinary/Documents/project/images'
IMG_SIZE = 200
namenum = 0
path = '/home/zinary/Documents/project/imagespro'
def label_img(img):
    word_label = img.split(" ")[0]
    return word_label
os.chdir(path)

for img in tqdm(os.listdir(DIR)):
    path = os.path.join(DIR,img)
    # edges = cv2.Canny(img,100,100)
    image = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),(IMG_SIZE,IMG_SIZE))
    name = label_img(img)
    namenum+=1
    dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    # blur = cv2.GaussianBlur(image,(5,5),0)
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite(name+str(namenum)+".jpg", thresh)

