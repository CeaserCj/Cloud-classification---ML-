import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dir="C:\\Users\\ragun\\OneDrive\\Desktop\\Untitled Folder"

categories = ['altostatus','cirrus','cumulonimbus','cumulus','nimbostratus']

data=[]

lables=[]

for category_idx, category in enumerate(categories):
   
    for file in os.listdir(os.path.join(dir,category)):
       
        img_path = os.path.join(dir, category, file)
       
        img = imread(img_path) 
        
        img = resize(img,(15,15))
        
        data.append(img.flatten())
        
        lables.append(category_idx)
        
data = np.asarray(data)

lables = np.asarray(lables)

x_train, x_test, y_train, y_test = train_test_split(data, lables, test_size=0.2, shuffle=True, Stratify=lables )

classifier = SVC()

parameters =[{'gama':[0.01,0.001,0.0001],'C':[1,10,100,1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of sample were correctly classified'.format(str(score*100)))
