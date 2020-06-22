'''
#Use of Support Vector Machine(SVM) Algorithm
dic1={'A':11,'B':21,'C':31}

print(dic1)
print(dic1["A"])

dic2={'A':[11,12,13,14],
      'B':[21,22,23,24,25],
      'C':[31,32,33]
      }

print(dic2["A"])

print(dic2.keys())

#print(dic2["A"])
#print(dic2.A)-this will not work

arr=['A','B','C','D','E','F','G','H']

print(list(enumerate(arr)))#List of Tuples
print(list(enumerate(arr[:4])))# First four values enumerated
print(list(enumerate(arr[:int(len(arr)/2)])))#Print first 50% of the data we can also use //2
print(list(enumerate(arr[int(len(arr)/2):])))#Print last 50% of data
print(list(enumerate((arr[int(len(arr)/2):])[:3])))#Print first 3 data from Last 50% of data ,also arr[n//2:n//2+3]
'''

import warnings
warnings.filterwarnings("ignore")

#Standard scientific Pyhton imports
import matplotlib.pyplot as plt

#Import datasets , classifiers and performance metrics
from sklearn import datasets,svm

#digits dataset
digits=datasets.load_digits()

print("digits:",digits.keys())
print("digits.target----: ",digits.target)

images_and_labels=list(zip(digits.images,digits.target))

print("len(images_and_labels):",len(images_and_labels))
for index,[image,label] in enumerate(images_and_labels[:5]):
    print("index:",index,"images:\n",image,"label:\n",label)
    plt.subplot(2,5,index+1)#positioning  numbering starts from 1
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Traning:%i'%label)
#if data is 2d then reshape into 1d so that we can get data of images in a row
#plt.show()

#To apply a classifier on the data, we need to flatten the image,
#turn the data in a (sample,features) matrix:
n_samples=len(digits.images)#=1797
print("n_samples:",n_samples)

imagedata=digits.images.reshape((n_samples,-1))

print("After Reshaped:len(imagedata[0]):",len(imagedata[0]))

#Create a classifier :a support vector classifier
classifier=svm.SVC(gamma=0.001)#SVC=support vector classifier

#We learn the digits on the first half of the digits
classifier.fit(imagedata[:n_samples//2],digits.target[:n_samples//2])

#Now predict the value of the digit on the second half
expectedy=digits.target[n_samples//2:]
predictedy=classifier.predict(imagedata[n_samples//2:])
images_and_prediction=list(zip(digits.images[n_samples//2:],predictedy))
for index,[image,prediction] in enumerate(images_and_prediction[:5]):
    plt.subplot(2,5,index+6)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')#interpolation=aas pass pe pixels clear dikhe
    plt.title('Prediction:%i'%prediction)

print("Original value:",digits.target[n_samples//2:n_samples//2+5])
#plt.show()

#Install Pillow Library
from scipy.misc import imread,imresize,bytescale
#import imageio

img=imread("THREE.jpg")
#img=imageio.imread("Three2.jpeg")
img=imresize(img,(8,8))
classifier=svm.SVC(gamma=0.001)
classifier.fit(imagedata[:],digits.target[:])

img=img.astype(digits.images.dtype)
img=bytescale(img,high=16.0,low=0)

print("img.shape:",img.shape)
print("\n",img)

x_testData=[]

for c in img:# Array of 8 values
    for r in c:# a line is made up of 3 colour
        x_testData.append(sum(r)/3.0)# take the average of all the three colour

print("x_testData:\n",x_testData)
print("len(x_testData):",len(x_testData))

x_testData=[x_testData]

print("Machine Output:",classifier.predict(x_testData))
plt.show()