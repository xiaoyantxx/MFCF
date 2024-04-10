# TMIF
A Multi-modal Fine-grained Classification Framework (MFCF) using deep learning for fine-grained fusulinid fossil recognition

### **The MFCF diagram**

![Uploading MFCF.png…](https://github.com/xiaoyantxx/MFCF/blob/main/MFCF_images/MFCF.png)

From the piper: Fine-grained Fusulinid Fossil Recognition by Combining Textual Information and Local Discriminative Visual 

### What is this repository for?

MFCF is a framework that uses deep learning for model training and fine-grained classification of fusulinid fossils thin-section.

### How do I get set up?

Install python 3.8 in the anaconda virtual environment on the Ubuntu operating system. Follow the requirements.txt file to configure the corresponding version of the package.

### Usage

1) Description of the document:

   data: storing OFM datasets for training

   dataManagement: it is responsible for separating the "class|image|text|" of a multi-modal dataset, processing and loading the dataset.

   multiManagement: it is responsible for obtaining the text vocabulary, and converting the images and text descriptions to encoding.

   models: the network file for MFCF. mdcf.py describes the overall network framework, and other files are called by it.

   runs: it holds the model trained by train.py.

   OFM：this folder holds the OFM dataset, where sample images are available for download.
   
   Arial.ttf: Font files. Used to print the predicted class name onto the image as a prediction display.

   train.py: for training optimal models.

   test.py: for testing.

   requirements.txt: includes a number of dependent packages.

3) Download the optimal training model trained by the authors. Click on the link: https://pan.baidu.com/s/1d-BLys-RbS4wxqzh72yy9A?pwd=srbi. The extraction code is srbi. Download the model and place it in the ./runs/weights folder.

5) Deploy the environment in IDE or terminal. Click test.py, set the weights file and the path to the multimodal dataset (already set). Click run to output the class predicted by the model.

6) Output file. In the ./runs/weights folder, output an acc.txt file with the categories of the model's predicted samples and the accuracy. The result images of the predicted samples will all be saved in the /runs/weights folder as well.

7) About OFM dataset: The OFM folder holds the multimodal dataset txt files in the format class|image|text, where all the image files are in the master branch. The dataset is divided into 45 classes. There are 9 classes with less than 2 samples, which cannot be classified for training, validation and test sets, so these 9 classes are removed for the experimental study. In subsequent studies will look for as many ways as possible to continue to collect the dataset in order to build a more robust model.

### Who do I talk to?

Fukai Zhang, Henan Polytechnic University

Email: zhangfukai@hpu.edu.cn
