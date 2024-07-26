import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Paths for image directory and model
# IMDIR=(sys.argv[1])
#IMDIR = '/home/rob/Pictures/Test_images/AT_test/'
#IMDIR = '/home/rob/Pictures/Rana/val/1/'
#IMDIR = '/home/rob/Pictures/Mite4Classes/finalDataSet1234/3/'

IMDIR = '/home/rob/Pictures/Tassles/test/Medium/'
#MODEL='models/resnet18.pth'
MODEL='models/tassle.pth'

# Load the model for testing
model = torch.load(MODEL, weights_only=False)
model.eval()

# Class labels for prediction
class_names=['DNA','High','Marginal', 'Medium']


files=Path(IMDIR).resolve().glob('*.*')

images = files # RJL select all the files in the folder for processing

# need to get file names out!
filenames = os.listdir(IMDIR) # RJL extracts file names to output later


# Preprocessing transformations
preprocess=transforms.Compose([
        #transforms.Resize(size=256),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

saveOutString = "DNA, High, Marginal, Medium, , ,prediction, filename"
saveOutString = saveOutString+'\n' # new line
myText = open(IMDIR+'my_text_file_softmax.csv',"w")

jpgCheck = 'jpg'
pngCheck = 'png'
imageNumber=0 # use own numbering in case it skips files to avoid out of bounds errors
with torch.no_grad():
    for num,img in enumerate(images):
      if pngCheck in img.name:
         img=Image.open(img).convert('RGB')
         inputs=preprocess(img).unsqueeze(0).to(device)
         outputs = model(inputs)

         _, preds = torch.max(outputs, 1)

         preds2 = torch.softmax(outputs, dim=-1, dtype=torch.float) # RJL outputs probabilities that add to 1? used when only 1 class is possible to choose
         preds3 = torch.sigmoid(outputs) # RJL outputs probabilities which don't have to add to 1, e.g. more than one possibility

         label=class_names[preds]

         #lineString = (str(preds3) + "@predicted_score=@" + label + "@" + filenames[imageNumber]) # using sigmoid output for probabilities
         lineString = (str(preds2) + "@predicted_score=@" + label + "@" + filenames[imageNumber]) # using softmax output for probabilities
         # remove bits from the string not needed for the csv file
         lineString = lineString.replace(" ", "")
         lineString = lineString.replace("(", "")
         lineString = lineString.replace(")", "")
         lineString = lineString.replace("[", "")
         lineString = lineString.replace("]", "")
         lineString = lineString.replace("@", ",")
         lineString = lineString.replace("tensor", "")
         lineString = lineString.replace("\n", ",")
         lineString = lineString.replace(",,", ",")
         lineString = lineString.replace("$", ",") # cope with file name conversion to x and y.
         print(lineString) # print to log as a check
         # add to text file on a new line
         saveOutString = saveOutString+lineString+'\n' # new line
         #iterate the next image number
         imageNumber = imageNumber+1

# save out txt file
myText.write(saveOutString)
myText.close()



# Perform prediction and plot results
#with torch.no_grad():
#    for num,img in enumerate(images):
 #        img=Image.open(img).convert('RGB')
  #       inputs=preprocess(img).unsqueeze(0).to(device)
   #      outputs = model(inputs)

         #print("outputs="+str(outputs)) # RJL output probabilities as raw numbers and parse into a string

    #     _, preds = torch.max(outputs, 1)
     #    label=class_names[preds]
         #plt.subplot(rows,cols,num+1)
         #plt.title("Pred: "+label)
      #   print(str(outputs[0])+"@predicted_score=@"+label+"@"+filenames[num])
         #plt.axis('off')
         #plt.imshow(img)
'''
Sample run: python test.py test
'''
