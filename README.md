This is code and dataset of paper "Unified meaning representation for guiding visual attention under diverse tasks in real-world scenes".
### Datasets

Download the datasets, and move the folders to ./data. The newly collected datasets in this paper:

* Dataset of THUE-task [[Download]](https://cloud.tsinghua.edu.cn/f/7e6307b843c840a9965f/?dl=1)
* Dataset of THUE-subway [[Download]](https://cloud.tsinghua.edu.cn/f/b359fba65e444a6594d1/?dl=1)

The existing datasets:

* DHF1K - https://github.com/wenguanwang/DHF1K
* Hollywood-2 - http://vision.imar.ro/eyetracking/description.php
* THUE-free - https://github.com/LiamXie/UrbanVisualAttention
* UCF-Sport - http://vision.imar.ro/eyetracking/description.php
* LEDOV - https://github.com/remega/LEDOV-eye-tracking-database
* DR(eye)VE - https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8

### **Dependencies**
* torch 2.5.1+cuda 12.1
* torchvision 0.20.1
* open-clip-torch 2.32.0
* opencv-python 4.11.0.86

### Test

After installing dependencies, you can run:

python test.py

This script will generate meaning map of the samples in ./data/test/. If the code works, results would be saved at ./outputs/test/

### Generating MRS

To generate MRS map for searching / wayfinding/ driving datasets, you can run :

python main_meaningmap_task.py

To generate MRS map for free-viewing datasets, you can run :

python main_meaningmap_free.py
