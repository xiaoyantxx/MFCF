# TMIF
A Multi-modal Fine-grained Classification Framework (MFCF) using deep learning for fine-grained fusulinid fossil recognition

### **The MFCF diagram**

![MFCF](C:\Users\闫正丽\Desktop\paper\paper2提交材料\Figure(png)\MFCF.png)

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
   
   Arial.ttf: Font files. Used to print the predicted class name onto the image as a prediction display.

   train.py: for training optimal models.

   test.py: for testing.

   requirements.txt: includes a number of dependent packages.

2) Download the optimal training model trained by the authors. Click on the link: https://pan.baidu.com/s/1d-BLys-RbS4wxqzh72yy9A?pwd=srbi. The extraction code is srbi. Download the model and place it in the ./runs/weights folder. 

3) Deploy the environment in IDE or terminal. Click test.py, set the weights file and the path to the multimodal dataset (already set). Click run to output the class predicted by the model.

4) Output file. In the ./runs/weights folder, output an acc.txt file with the categories of the model's predicted samples and the accuracy. The result images of the predicted samples will all be saved in the /runs/weights folder as well.

### Who do I talk to?

Fukai Zhang, Henan Polytechnic University

Email: zhangfukai@hpu.edu.cn
