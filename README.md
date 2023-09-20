# YOLO-Shape-Detection
Welcome to the YOLO-Shape-Detection repository! This project showcases an implementation of the YOLO (You Only Look Once) object detection algorithm tailored specifically for detecting and classifying shapes within images.


# Gpu:
Let's make sure that we have access to GPU. We can use nvidia-smi command to do that. In case of any problems navigate to Edit -> Notebook settings -> Hardware accelerator, set it to GPU, and then click Save.

```python
!nvidia-smi
```

# import os:
 This line imports the Python os module, which provides a way to interact with the operating system, including functions for working with files and directories.

# HOME = os.getcwd():
 This line assigns the result of the os.getcwd() function to a variable named HOME. os.getcwd() stands for "get current working directory," and it returns the current directory path where the Python script is being executed.

# print(HOME):
 This line prints the value of the HOME variable, which is the current working directory, to the console or standard output.

```python
import os
HOME = os.getcwd()
print(HOME)
```

# !pip install ultralytics==8.0.20:
This line installs the ultralytics library with version 8.0.20 using the pip package manager. The ultralytics library is commonly used for computer vision and deep learning tasks, including object detection and image classification.

# from IPython import display:
 This line imports the display module from the IPython library. IPython is an interactive Python environment, and display can be used to control the display of output within IPython notebooks.

# display.clear_output():
 This line clears the current output in the IPython notebook. It's often used to keep the notebook clean and display only the relevant information.

# import ultralytics:
 This imports the ultralytics library that you installed earlier.

# ultralytics.checks():
This is a call to the checks() function provided by the ultralytics library. It's used to perform checks on the library's installation and dependencies. It ensures that everything is set up correctly for you to use ultralytics for your computer vision tasks.

```python
!pip install ultralytics==8.0.20
from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
```

# from ultralytics import YOLO:
 This imports the YOLO class from the ultralytics library. The YOLO class is typically used for object detection tasks, and ultralytics provides an easy-to-use interface for training and evaluating YOLO models.

# from IPython.display import display, Image:
 These lines import the display function and the Image class from the IPython.display module. These functions and classes allow you to display images and other content directly within an IPython notebook, making it convenient for visualizing the results of your object detection tasks.

```python
 from ultralytics import YOLO
from IPython.display import display, Image
```

command that changes the current directory to the path stored in the HOME variable. This command ensures that you are working in the correct directory.

# task=detect:
Specifies that the task is object detection.

# mode=predict:
Sets the mode to prediction (inference) mode.

# model=yolov8n.pt:

Specifies the YOLOv8 model to be used for detection. It's loading a model file named yolov8n.pt which is a pre-trained model just to check we are working fine till now.


# conf=0.25:

Sets the confidence threshold for detection. Objects with a confidence score greater than or equal to 0.25 will be considered as valid detections.


# source='anyimagepath':

Specifies the source for the input image. In this case, it's a URL pointing to an image of a dog.


# save=True:

 Indicates that the results of the detection task should be saved.

```python
 %cd {HOME}
!yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=True
```

# Image(filename='runs/detect/predict/dog.jpeg', height=600):

This line displays an image located at the specified file path 'runs/detect/predict/dog.jpeg' with a specified height of 600 pixels.

 Image is a class from the IPython.display module that allows you to display images and other content directly within an IPython notebook.
filename is the parameter that specifies the file path of the image you want to display.

# height=600

is an optional parameter that sets the height of the displayed image to 600 pixels. You can adjust this value to change the image's displayed size.


```python
%cd {HOME}
Image(filename='runs/detect/predict/dog.jpeg', height=600)
```
![download](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/a9c40e1e-b426-48c5-889a-a774ca9d9a9d)

```python
model = YOLO(f'{HOME}/yolov8n.pt')
results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)
```

```python
results[0].boxes.xyxy
```
```txt
tensor([[   0.,  314.,  625., 1278.],
        [  55.,  250.,  648., 1266.],
        [ 633.,  720.,  701.,  786.]], device='cuda:0')
```

```python
results[0].boxes.conf
```

```txt
tensor([0.72713, 0.29066, 0.28456], device='cuda:0')
```

```python
results[0].boxes.cls
```
```txt
tensor([ 0., 16.,  2.], device='cuda:0')
```

#Now for our own dataset of geometric shapes:

##classes:


1.   CIRCLE
2.   CROSS


3.   HEPTAGON
4.   OCTAGON

5.  HEXAGON

6.  PENTAGON

7.  QUARTER_CIRCLE

8.  RECTANGLE

9.  SEMICIRCLE

10. SQUARE
11. STAR
12. TRAPEZOID
13. TRIANGLE



##YOLO VERSION:
used yolo version 5 which is compatible with the dataset type downloaded from Roboflow.


```python
!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow --quiet

from roboflow import Roboflow
rf = Roboflow(api_key="zhmTBFaVxqAy7nfb9N3j")
project = rf.workspace("hku-uas-deprecated-sobt2").project("standard_object_shape")
dataset = project.version(2).download("yolov5")
```

# now let's start training for 5 epochs:
```python
%cd {HOME}

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=5 imgsz=800 plots=True
```
# @ epoch 5 those are the results:
```txt
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        5/5      7.25G      1.036     0.9845     0.9738          3        800: 100% 1224/1224 [17:17<00:00,  1.18it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 59/59 [00:42<00:00,  1.38it/s]
                   all       1864       1864      0.818      0.902      0.911      0.687
                CIRCLE       1864        167      0.491      0.976      0.866      0.659
                 CROSS       1864        127       0.94      0.981      0.992      0.776
              HEPTAGON       1864        137      0.594      0.651      0.724      0.546
               HEXAGON       1864        142      0.662      0.838       0.88      0.672
               OCTAGON       1864        147      0.422      0.544       0.47      0.345
              PENTAGON       1864        138       0.94      0.949      0.985      0.738
        QUARTER_CIRCLE       1864        141      0.904      0.972      0.988      0.755
             RECTANGLE       1864        143      0.933      0.986      0.993      0.781
            SEMICIRCLE       1864        160      0.936      0.956      0.988      0.739
                SQUARE       1864        133      0.975      0.947      0.989      0.783
                  STAR       1864        149      0.959      0.993      0.993      0.721
             TRAPEZOID       1864        142      0.914      0.972      0.989      0.771

```
## now let's see what files did we generate after learining:
```python
!ls {HOME}/runs/detect/train/
```

```txt
args.yaml					    train_batch1.jpg
confusion_matrix.png				    train_batch2.jpg
events.out.tfevents.1695185683.f9fe2eb31fa8.2355.0  val_batch0_labels.jpg
F1_curve.png					    val_batch0_pred.jpg
P_curve.png					    val_batch1_labels.jpg
PR_curve.png					    val_batch1_pred.jpg
R_curve.png					    val_batch2_labels.jpg
results.csv					    val_batch2_pred.jpg
results.png					    weights
train_batch0.jpg
```
# confusioon matrix:
```python
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
```

![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/c7588c28-6607-42fb-9dcf-a59247ff7822)
## results:
```python
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)
```
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/2332c864-2f8b-4dfb-ab9a-fe5d631a7403)

```python
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)
```
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/75b55e02-2519-4490-9a17-774fafaf66b4)


```python
%cd {HOME}

!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
```



```txt
/content
2023-09-20 06:28:01.899474: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Ultralytics YOLOv8.0.20 🚀 Python-3.10.12 torch-2.0.1+cu118 CUDA:0 (Tesla T4, 15102MiB)
Model summary (fused): 168 layers, 11130615 parameters, 0 gradients, 28.5 GFLOPs
val: Scanning /content/datasets/standard_object_shape-2/valid/labels.cache... 1864 images, 0 backgrounds, 0 corrupt: 100% 1864/1864 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 117/117 [00:45<00:00,  2.57it/s]
                   all       1864       1864      0.818      0.902      0.911       0.69
                CIRCLE       1864        167       0.49      0.976      0.866      0.659
                 CROSS       1864        127       0.94      0.981      0.992      0.779
              HEPTAGON       1864        137      0.595      0.653      0.723      0.547
               HEXAGON       1864        142       0.66      0.838      0.879      0.676
               OCTAGON       1864        147      0.418      0.544       0.47       0.35
              PENTAGON       1864        138       0.94      0.949      0.985      0.737
        QUARTER_CIRCLE       1864        141      0.903      0.972      0.988      0.754
             RECTANGLE       1864        143      0.932      0.986      0.993      0.786
            SEMICIRCLE       1864        160      0.935      0.956      0.988      0.742
                SQUARE       1864        133      0.974      0.947      0.989      0.788
                  STAR       1864        149      0.959      0.993      0.993      0.722
             TRAPEZOID       1864        142      0.913      0.972      0.989      0.775
              TRIANGLE       1864        138      0.969      0.964      0.988      0.656
Speed: 1.0ms pre-process, 13.2ms inference, 0.0ms loss, 2.0ms post-process per image
```
```python
%cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
```

```python
import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict2/*.jpg')[10:25]:
      display(Image(filename=image_path, width=600))
      print("\n")
```

![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/65b41256-de35-48af-9ea2-3cbbd722de13)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/258adf86-938d-4800-80cf-7e95d2d3df6e)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/b0fe4e22-21ec-4805-a7d0-a4691e26a696)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/592d8e88-11ac-410d-a730-98d86fecba45)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/2c8d68be-db06-4d1a-af24-1d1489013fad)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/c2a538d1-25d2-4c7d-a537-7b6d3ad8d658)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/633bb029-d6db-43ee-94cc-0a38f3553e6a)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/700c4f47-8223-4dda-9b10-f2383d7be621)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/117369db-5835-4c20-8334-52bc6bb1ba3b)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/213f25f2-828b-4bc1-8c20-1e947808f35a)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/cdb1af91-f3ae-410f-b32e-97f9769a89f1)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/3886a1e8-3e8a-4b75-b631-c12b63f807e7)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/f33b8121-9163-4c45-a505-fc1206570341)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/b2ff4127-8c68-40a8-8c9e-f6840cbcf60e)
![image](https://github.com/Omarmahmoud711/YOLO-Shape-Detection/assets/128754061/88ab45f0-fac1-44e0-88d6-fabf4617ef45)









