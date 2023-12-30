#importing project
import pickle

from img2vec_pytorch import Img2Vec

from PIL import Image

with open('.\\project.p', 'rb') as f:
    
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = "C:\\Users\\ragun\\OneDrive\\Desktop\\Untitled Folder\\deepan.jpg"

img = Image.open(image_path)

features = img2vec.get_vec(img)

pred = model.predict([features])

print(pred)

