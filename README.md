# Object Detection using YOLO-v3 Deep Learning Model Trained on COCO Dataset in Python

<img src="images/YOLO-v3-003.jpg" width = "1000"/>

## 1. Objective

In this project, we shall demonstrate the Deep Learning (DL) inference using a DL object detection model, YOLO-v3, which has already been trained on the COCO dataset. 

## 2. YOLO-v3

* You Only Look Once (YOLO) is a state-of-the-art, real-time object detection model. It is not straight forward to reasonably train the YOLO-v3 network from scratch, due to several reasons including: 

  * Lack of large volume of annotated data 
  * Lack of sufficiently powerful computing resources 

* Instead of exploring the training of YOLO-v3 from scratch we use an already trained model retrieved from the following source: 

  * Trained YOLO-v3 model source: https://github.com/xiaochus/YOLOv3 
  * This model has been trained on the COCO dataset: 
    * Source: https://cocodataset.org/#home 
  * COCO is a large-scale object detection, segmentation, and captioning dataset. 
  * 330K images (>200K labeled) 
  * 80 object categories, including typical objects such as vehicles, people, cats, dogs, etc. 

In this work, we shall demonstrate how to deploy the trained YOLO-v3 model to detect objects of interest:

## 3. Development

* Project: Object Detection using YOLO-v3:
  * The objective of this project is to demonstrate how to use the state of the art in object detection!

* It is not straight forward to reasonably train the YOLO-v3 network from scratch, due to several reasons including:
  * Lack of large volume of annotated data
  * Lack of sufficiently powerful computing resources
  * Instead of exploring the training of YOLO-v3 from scratch we use an already trained model retrieved from the following source:
    * Trained YOLO-v3 model source: https://github.com/xiaochus/YOLOv3
    * Original YOLO-v3 paper: YOLOv3: An Incremental Improvement
      * Source: https://arxiv.org/abs/1804.02767
    * This model has been trained on the COCO dataset:
    * Source: https://cocodataset.org/#home
      * COCO is a large-scale object detection, segmentation, and captioning dataset.
        * 330K images (>200K labeled)
        * 80 object categories, including typical objects such as vehicles, people, cats, dogs, etc.

* In this work, we shall demonstrate how to deploy the trained model to detect objects of interest:
  * Author: Mohsen Ghazel (mghazel)
  * Date: April 9th, 2021
  
### 3.1. Step 1: Imports and global variables

#### 3.1.1. Python import

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>image <span style="color:#200080; font-weight:bold; ">as</span> mpimg

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Import the YOLO model from the downlaoded</span>
<span style="color:#595979; "># pretraiend model subfolder (model)</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#200080; font-weight:bold; ">from</span> model<span style="color:#308080; ">.</span>yolo_model <span style="color:#200080; font-weight:bold; ">import</span> YOLO

<span style="color:#595979; "># input/output OS</span>
<span style="color:#200080; font-weight:bold; ">import</span> os 

<span style="color:#595979; "># date-time to show date and time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime
<span style="color:#595979; "># import time</span>
<span style="color:#200080; font-weight:bold; ">import</span> time

<span style="color:#595979; "># to display the figures in the notebook</span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">4.5</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">1</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>

#### 3.1.2. Global variables:

* The YOLO model expects two parameters:
  * Confidence threshold: Only objects with detection confidence higher than this value are kept and the rest are discarded
  * NMS threshold: This is used by the NMS algorithm to combine overlapping detections.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Confidence threshold: </span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Only objects with detection confidence higher than this </span>
<span style="color:#595979; "># value are kept and the rest are discarded</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
DETECTION_CONFIDENCE_THRESHOLD <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.50</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># NMS threshold: </span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># This is used by the NMS algorithm to combine overlapping </span>
<span style="color:#595979; "># detections</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
NMS_THRESHOLD <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.50</span>
</pre>

### 3.2. Step 2: Define utility functions:
#### 3.2.1. A function pre-process images by resizing them as needed:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> process_image<span style="color:#308080; ">(</span>img<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""Resize, reduce and expand image.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Argument:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img: original image.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Returns</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image: ndarray(64, 64, 3), processed image.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    image <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>resize<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">416</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">416</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
                       interpolation<span style="color:#308080; ">=</span>cv2<span style="color:#308080; ">.</span>INTER_CUBIC<span style="color:#308080; ">)</span>
    image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>array<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> dtype<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'float32'</span><span style="color:#308080; ">)</span>
    image <span style="color:#44aadd; ">/</span><span style="color:#308080; ">=</span> <span style="color:#008000; ">255.</span>
    image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>expand_dims<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> axis<span style="color:#308080; ">=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>

    <span style="color:#200080; font-weight:bold; ">return</span> image
</pre>


#### 3.2.2. A function to get the class names:

* The YOLO-v3 model has been pre-trained on the COCO dataset, which has 80 classes
* This function retrieves the names of the 80 classes in the COCO data set by reading them from the configuration file

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> get_classes<span style="color:#308080; ">(</span><span style="color:#400000; ">file</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""Get classes name.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Argument:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;file: classes name for database.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Returns</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class_names: List, classes name.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#200080; font-weight:bold; ">with</span> <span style="color:#400000; ">open</span><span style="color:#308080; ">(</span><span style="color:#400000; ">file</span><span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">as</span> f<span style="color:#308080; ">:</span>
        class_names <span style="color:#308080; ">=</span> f<span style="color:#308080; ">.</span>readlines<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    class_names <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span>c<span style="color:#308080; ">.</span>strip<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">for</span> c <span style="color:#200080; font-weight:bold; ">in</span> class_names<span style="color:#308080; ">]</span>

    <span style="color:#200080; font-weight:bold; ">return</span> class_names
</pre>

#### 3.2.3. A visualization function:
* This function overlays the detection results on this image, including:
  * detection bounding-boxes
  * class name
  * detection confidence

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> draw<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> boxes<span style="color:#308080; ">,</span> scores<span style="color:#308080; ">,</span> classes<span style="color:#308080; ">,</span> all_classes<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""Draw the boxes on the image.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Argument:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image: original image.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;boxes: ndarray, boxes of objects.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;classes: ndarray, classes of objects.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scores: ndarray, scores of objects.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;all_classes: all classes name.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#200080; font-weight:bold; ">for</span> box<span style="color:#308080; ">,</span> score<span style="color:#308080; ">,</span> cl <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">zip</span><span style="color:#308080; ">(</span>boxes<span style="color:#308080; ">,</span> scores<span style="color:#308080; ">,</span> classes<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        x<span style="color:#308080; ">,</span> y<span style="color:#308080; ">,</span> w<span style="color:#308080; ">,</span> h <span style="color:#308080; ">=</span> box

        top <span style="color:#308080; ">=</span> <span style="color:#400000; ">max</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>floor<span style="color:#308080; ">(</span>x <span style="color:#44aadd; ">+</span> <span style="color:#008000; ">0.5</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span><span style="color:#400000; ">int</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        left <span style="color:#308080; ">=</span> <span style="color:#400000; ">max</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>floor<span style="color:#308080; ">(</span>y <span style="color:#44aadd; ">+</span> <span style="color:#008000; ">0.5</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span><span style="color:#400000; ">int</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        right <span style="color:#308080; ">=</span> <span style="color:#400000; ">min</span><span style="color:#308080; ">(</span>image<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>floor<span style="color:#308080; ">(</span>x <span style="color:#44aadd; ">+</span> w <span style="color:#44aadd; ">+</span> <span style="color:#008000; ">0.5</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span><span style="color:#400000; ">int</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        bottom <span style="color:#308080; ">=</span> <span style="color:#400000; ">min</span><span style="color:#308080; ">(</span>image<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>floor<span style="color:#308080; ">(</span>y <span style="color:#44aadd; ">+</span> h <span style="color:#44aadd; ">+</span> <span style="color:#008000; ">0.5</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span><span style="color:#400000; ">int</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

        cv2<span style="color:#308080; ">.</span>rectangle<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>top<span style="color:#308080; ">,</span> left<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>right<span style="color:#308080; ">,</span> bottom<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span>
        cv2<span style="color:#308080; ">.</span>putText<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">'{0} {1:.2f}'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>all_classes<span style="color:#308080; ">[</span>cl<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> score<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
                    <span style="color:#308080; ">(</span>top<span style="color:#308080; ">,</span> left <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">6</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
                    cv2<span style="color:#308080; ">.</span>FONT_HERSHEY_SIMPLEX<span style="color:#308080; ">,</span>
                    <span style="color:#008000; ">0.6</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span>
                    cv2<span style="color:#308080; ">.</span>LINE_AA<span style="color:#308080; ">)</span>

        <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'class: {0}, score: {1:.2f}'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>all_classes<span style="color:#308080; ">[</span>cl<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> score<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'box coordinate x,y,w,h: {0}'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>box<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

#### 3.2.4. Inference function from an image:

* This function deploys the trained YOLO-v3 model to detect any potential objects of interest.

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> detect_image<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> yolo<span style="color:#308080; ">,</span> all_classes<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""Use yolo v3 to detect images.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Argument:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image: original image.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;yolo: YOLO, yolo model.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;all_classes: all classes name.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image: processed image.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    pimage <span style="color:#308080; ">=</span> process_image<span style="color:#308080; ">(</span>image<span style="color:#308080; ">)</span>

    start <span style="color:#308080; ">=</span> time<span style="color:#308080; ">.</span>time<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    boxes<span style="color:#308080; ">,</span> classes<span style="color:#308080; ">,</span> scores <span style="color:#308080; ">=</span> yolo<span style="color:#308080; ">.</span>predict<span style="color:#308080; ">(</span>pimage<span style="color:#308080; ">,</span> image<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    end <span style="color:#308080; ">=</span> time<span style="color:#308080; ">.</span>time<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'time: {0:.2f}s'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>end <span style="color:#44aadd; ">-</span> start<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

    <span style="color:#200080; font-weight:bold; ">if</span> boxes <span style="color:#200080; font-weight:bold; ">is</span> <span style="color:#200080; font-weight:bold; ">not</span> <span style="color:#074726; ">None</span><span style="color:#308080; ">:</span>
        draw<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> boxes<span style="color:#308080; ">,</span> scores<span style="color:#308080; ">,</span> classes<span style="color:#308080; ">,</span> all_classes<span style="color:#308080; ">)</span>

    <span style="color:#200080; font-weight:bold; ">return</span> image
</pre>

#### 3.2.5. Inference function from a video:

* This function deploys the trained YOLO-v3 model to detect any potential objects of interest from a video file
* Basically, each video frame is processes independently, using the previous function.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> detect_video<span style="color:#308080; ">(</span>video<span style="color:#308080; ">,</span> yolo<span style="color:#308080; ">,</span> all_classes<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""Use yolo v3 to detect video.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Argument:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;video: video file.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;yolo: YOLO, yolo model.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;all_classes: all classes name.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    video_path <span style="color:#308080; ">=</span> os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>join<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"videos"</span><span style="color:#308080; ">,</span> <span style="color:#1060b6; ">"test"</span><span style="color:#308080; ">,</span> video<span style="color:#308080; ">)</span>
    camera <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span>video_path<span style="color:#308080; ">)</span>
    cv2<span style="color:#308080; ">.</span>namedWindow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"detection"</span><span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>WINDOW_AUTOSIZE<span style="color:#308080; ">)</span>

    <span style="color:#595979; "># Prepare for saving the detected video</span>
    sz <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#400000; ">int</span><span style="color:#308080; ">(</span>camera<span style="color:#308080; ">.</span>get<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>CAP_PROP_FRAME_WIDTH<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
        <span style="color:#400000; ">int</span><span style="color:#308080; ">(</span>camera<span style="color:#308080; ">.</span>get<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>CAP_PROP_FRAME_HEIGHT<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    fourcc <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoWriter_fourcc<span style="color:#308080; ">(</span><span style="color:#44aadd; ">*</span><span style="color:#1060b6; ">'mpeg'</span><span style="color:#308080; ">)</span>

    
    vout <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoWriter<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    vout<span style="color:#308080; ">.</span><span style="color:#400000; ">open</span><span style="color:#308080; ">(</span>os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>join<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"videos"</span><span style="color:#308080; ">,</span> <span style="color:#1060b6; ">"res"</span><span style="color:#308080; ">,</span> video<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> fourcc<span style="color:#308080; ">,</span> <span style="color:#008c00; ">20</span><span style="color:#308080; ">,</span> sz<span style="color:#308080; ">,</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>

    <span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
        res<span style="color:#308080; ">,</span> frame <span style="color:#308080; ">=</span> camera<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

        <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> res<span style="color:#308080; ">:</span>
            <span style="color:#200080; font-weight:bold; ">break</span>

        image <span style="color:#308080; ">=</span> detect_image<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> yolo<span style="color:#308080; ">,</span> all_classes<span style="color:#308080; ">)</span>
        cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"detection"</span><span style="color:#308080; ">,</span> image<span style="color:#308080; ">)</span>

        <span style="color:#595979; "># Save the video frame by frame</span>
        vout<span style="color:#308080; ">.</span>write<span style="color:#308080; ">(</span>image<span style="color:#308080; ">)</span>

        <span style="color:#200080; font-weight:bold; ">if</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">110</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#308080; ">:</span>
                <span style="color:#200080; font-weight:bold; ">break</span>

    vout<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    camera<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    
</pre>

### 3.3. Step 3: Instantiate the YOLO-v3 model:
* We are now ready to instantiate the pre-trained YOLO-v3 model before it can be deployed:
  * The YOLO model expects two parameters:
    * Confidence threshold: Only objects with detection confidence higher than this value are kept and the rest are discarded
    * NMS threshold: This is used by the NMS algorithm to combine overlapping detections.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Instantiate the YOLO model class</span>
yolo <span style="color:#308080; ">=</span> YOLO<span style="color:#308080; ">(</span>DETECTION_CONFIDENCE_THRESHOLD<span style="color:#308080; ">,</span> NMS_THRESHOLD<span style="color:#308080; ">)</span>
<span style="color:#595979; "># this is the file containing the names of all the classes</span>
<span style="color:#400000; ">file</span> <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'data/coco_classes.txt'</span>
<span style="color:#595979; "># get all the names of the classes</span>
all_classes <span style="color:#308080; ">=</span> get_classes<span style="color:#308080; ">(</span><span style="color:#400000; ">file</span><span style="color:#308080; ">)</span>
</pre>


### 3.4. Step 4: Deploy the trained YOLO-v3 model to detect objects of interest:
#### 3.4.1. Detection from images:
* First we can deploy it on individual input images to detect and localize potential objects of interest
  * All test images in the test images folder will be processed
  * We can get test images from open source site:
  * Unsplash: https://unsplash.com/


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.1) Set test images folder name</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
test_images_folder <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'images/test/'</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.2) itetate over all the images in the test images </span>
<span style="color:#595979; ">#      folder</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#200080; font-weight:bold; ">for</span> filename <span style="color:#200080; font-weight:bold; ">in</span> os<span style="color:#308080; ">.</span>listdir<span style="color:#308080; ">(</span>test_images_folder<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># 4.3) read the test image</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    img <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>imread<span style="color:#308080; ">(</span>os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>join<span style="color:#308080; ">(</span>test_images_folder<span style="color:#308080; ">,</span>filename<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> img <span style="color:#200080; font-weight:bold; ">is</span> <span style="color:#200080; font-weight:bold; ">not</span> <span style="color:#074726; ">None</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        <span style="color:#595979; "># 4.4) deploy the YOLO model to conduct inference in </span>
        <span style="color:#595979; ">#      the image</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        img <span style="color:#308080; ">=</span> detect_image<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> yolo<span style="color:#308080; ">,</span> all_classes<span style="color:#308080; ">)</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        <span style="color:#595979; "># 4.5) Visualize the detections results</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        <span style="color:#595979; "># create a figure</span>
        plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>uint8<span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span> <span style="color:#44aadd; ">*</span> img<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span>img<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># visualize detection results</span>
        plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">111</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"YOLO-v3: Detection results"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>img<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2RGB<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
</pre>

#### 3.4.12. Detection from videos:
* We can deploy the trained YOLO-v3 model on a video to detect and localize potential objects of interest:
  * Basically, each frame of the video is parsed and processed separately as an image, just like we illustrated above.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># detect videos one at a time in videos/test folder   </span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.2.1) Set test video folder and file name</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># test-video folder</span>
test_video_folder <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'images/test/'</span>
<span style="color:#595979; "># test-video file name</span>
test_video_fname <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'library1.mp4'</span>
<span style="color:#595979; "># test-video full-path file name</span>
test_video_file_path <span style="color:#308080; ">=</span> test_video_folder <span style="color:#44aadd; ">+</span> test_video_fname 
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.2.2) call detect_video() to conduct inference on </span>
<span style="color:#595979; ">#         the video</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
detect_video<span style="color:#308080; ">(</span>test_video_file_path<span style="color:#308080; ">,</span> yolo<span style="color:#308080; ">,</span> all_classes<span style="color:#308080; ">)</span>
</pre>

### 3.4. Step 4: Display a successful execution message:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">14</span> <span style="color:#008c00; ">21</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">39</span><span style="color:#308080; ">:</span><span style="color:#008000; ">22.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>


## 4. Sample YOLO object detection results

* We have tested the YOLO-v3 model on 10 test images, contains various objects of interest. The detection results are illustrated next for each test image.

### 4.1. Test image # 1:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">1.78</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> dog<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.99</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">10.19209296</span>  <span style="color:#008000; ">14.36614296</span>  <span style="color:#008000; ">86.29040748</span> <span style="color:#008000; ">132.3982138</span> <span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> dog<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.95</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">126.09061539</span>   <span style="color:#008000; ">6.01810643</span>  <span style="color:#008000; ">74.22350317</span> <span style="color:#008000; ">146.36979657</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> dog<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.56</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">81.35105833</span> <span style="color:#008000; ">62.04934654</span> <span style="color:#008000; ">61.20964855</span> <span style="color:#008000; ">82.47159237</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> cat<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.53</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">198.27343404</span>  <span style="color:#008000; ">63.17662293</span>  <span style="color:#008000; ">53.06311816</span>  <span style="color:#008000; ">79.39320123</span><span style="color:#308080; ">]</span>
</pre>

<img src="test-image-01-results.webp" width = "1000"/>


### 4.2. Test image # 2:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.74</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.99</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">368.21359396</span> <span style="color:#008000; ">104.02029443</span> <span style="color:#008000; ">121.8848601</span>  <span style="color:#008000; ">120.1177237</span> <span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.97</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">50.01873523</span> <span style="color:#008000; ">111.84529734</span> <span style="color:#008000; ">139.93528485</span> <span style="color:#008000; ">141.13240802</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.83</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">36.13579646</span>  <span style="color:#008000; ">51.83125204</span>  <span style="color:#008000; ">62.23230809</span> <span style="color:#008000; ">158.29805338</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.62</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">166.49897397</span>  <span style="color:#008000; ">66.85848141</span>  <span style="color:#008000; ">58.9142479</span>  <span style="color:#008000; ">106.41536546</span><span style="color:#308080; ">]</span>
</pre>

<img src="test-image-02-results.webp" width = "1000"/>

### 4.3 Test image # 3:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.74</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.99</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">378.28749418</span> <span style="color:#008000; ">326.94270372</span> <span style="color:#008000; ">121.24058604</span> <span style="color:#008000; ">305.88562575</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.98</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">194.84049082</span> <span style="color:#008000; ">341.5541026</span>  <span style="color:#008000; ">109.87009108</span> <span style="color:#008000; ">291.49284735</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.97</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">2.07313523</span> <span style="color:#008000; ">329.88191319</span> <span style="color:#008000; ">111.28226668</span> <span style="color:#008000; ">296.5332661</span> <span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.97</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">99.36891496</span> <span style="color:#008000; ">326.99353135</span> <span style="color:#008000; ">117.07406491</span> <span style="color:#008000; ">303.14106911</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.70</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">281.08209372</span> <span style="color:#008000; ">328.79640794</span> <span style="color:#008000; ">151.02022886</span> <span style="color:#008000; ">306.41904584</span><span style="color:#308080; ">]</span>
</pre>

<img src="test-image-03-results.webp" width = "1000"/>



### 4.4. Test image # 4:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.80</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> horse<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.99</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">68.49596649</span>  <span style="color:#008000; ">97.66224182</span> <span style="color:#008000; ">116.75174534</span> <span style="color:#008000; ">193.6813184</span> <span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> horse<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.98</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">259.71484184</span> <span style="color:#008000; ">138.50968283</span>  <span style="color:#008000; ">81.99092746</span> <span style="color:#008000; ">119.73992506</span><span style="color:#308080; ">]</span>
</pre>


<img src="test-image-04-results.webp" width = "1000"/>


### 4.5. Test image # 5:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.82</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> horse<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.99</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">2.93928385</span>  <span style="color:#008000; ">85.70314649</span> <span style="color:#008000; ">113.22645098</span> <span style="color:#008000; ">148.47420788</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> horse<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.99</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">230.75139523</span>  <span style="color:#008000; ">75.36754668</span> <span style="color:#008000; ">262.8532052</span>  <span style="color:#008000; ">250.40971935</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> horse<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.96</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">152.32335031</span>  <span style="color:#008000; ">79.1682272</span>  <span style="color:#008000; ">130.95849752</span> <span style="color:#008000; ">242.19710863</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> horse<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.90</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">88.25491369</span> <span style="color:#008000; ">101.32178074</span> <span style="color:#008000; ">106.2919721</span>  <span style="color:#008000; ">194.65964341</span><span style="color:#308080; ">]</span>
</pre>

<img src="test-image-05-results.webp" width = "1000"/>



### 4.6. Test image # 6:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.71</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.99</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">287.1645689</span>  <span style="color:#008000; ">482.89136589</span>  <span style="color:#008000; ">90.47405422</span> <span style="color:#008000; ">220.44807673</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> car<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.59</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">61.97722256</span> <span style="color:#008000; ">494.72030997</span> <span style="color:#008000; ">128.77330184</span>  <span style="color:#008000; ">70.26270591</span><span style="color:#308080; ">]</span>
</pre>



<img src="test-image-06-results.webp" width = "1000"/>


### 4.7. Test image # 7:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.77</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">1.00</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">187.65594482</span>  <span style="color:#008000; ">82.93733287</span>  <span style="color:#008000; ">91.79075241</span> <span style="color:#008000; ">306.85243368</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> horse<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">1.00</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">396.35887146</span> <span style="color:#008000; ">137.40119696</span> <span style="color:#008000; ">215.85384369</span> <span style="color:#008000; ">208.32020402</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> dog<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">1.00</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">61.19189739</span> <span style="color:#008000; ">263.38934135</span> <span style="color:#008000; ">145.36686897</span>  <span style="color:#008000; ">88.4316597</span> <span style="color:#308080; ">]</span>
</pre>


<img src="test-image-07-results.webp" width = "1000"/>


### 4.8. Test image # 8:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.73</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.98</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">225.50240159</span> <span style="color:#008000; ">102.00662017</span>  <span style="color:#008000; ">77.20900327</span> <span style="color:#008000; ">151.62935683</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.89</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">106.22136295</span> <span style="color:#008000; ">108.16646779</span>  <span style="color:#008000; ">66.88002497</span> <span style="color:#008000; ">155.51600531</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> bicycle<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">1.00</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">214.48674798</span> <span style="color:#008000; ">169.38857824</span> <span style="color:#008000; ">126.11605227</span> <span style="color:#008000; ">115.49647996</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> bicycle<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.77</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">84.21652764</span> <span style="color:#008000; ">172.78498912</span> <span style="color:#008000; ">110.63914001</span>  <span style="color:#008000; ">97.43866879</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> car<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">1.00</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">343.9591229</span>  <span style="color:#008000; ">127.53708968</span> <span style="color:#008000; ">152.92595327</span>  <span style="color:#008000; ">77.33121413</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> car<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.95</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">48.66480827</span> <span style="color:#008000; ">126.89756724</span> <span style="color:#008000; ">172.94928432</span>  <span style="color:#008000; ">93.92557389</span><span style="color:#308080; ">]</span>
</pre>

<img src="test-image-08-results.webp" width = "1000"/>


### 4.9. Test image # 9:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.86</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.96</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">349.51111674</span> <span style="color:#008000; ">179.73068959</span> <span style="color:#008000; ">108.53108019</span> <span style="color:#008000; ">156.0712916</span> <span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.88</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">35.07959098</span> <span style="color:#008000; ">179.21235001</span>  <span style="color:#008000; ">60.12298167</span> <span style="color:#008000; ">146.15605992</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> person<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.54</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span> <span style="color:#008000; ">79.5147121</span>  <span style="color:#008000; ">180.1177125</span>   <span style="color:#008000; ">49.46985096</span> <span style="color:#008000; ">134.8184858</span> <span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> bicycle<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.96</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">160.18867493</span> <span style="color:#008000; ">215.96660542</span> <span style="color:#008000; ">106.28129542</span>  <span style="color:#008000; ">61.67598841</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> car<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.95</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">130.0984025</span>  <span style="color:#008000; ">178.52714539</span> <span style="color:#008000; ">156.98941052</span>  <span style="color:#008000; ">49.25055121</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> car<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.91</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">334.85984802</span> <span style="color:#008000; ">181.99032462</span> <span style="color:#008000; ">147.12804556</span>  <span style="color:#008000; ">41.20300182</span><span style="color:#308080; ">]</span>
</pre>

<img src="test-image-09-results.webp" width = "1000"/>


### 4.10. Test image # 10:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">time<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.77</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> bicycle<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.54</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">271.56883478</span> <span style="color:#008000; ">474.45206344</span> <span style="color:#008000; ">101.01432353</span>  <span style="color:#008000; ">58.80256556</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> car<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.98</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">362.14998364</span> <span style="color:#008000; ">348.10010344</span> <span style="color:#008000; ">129.40327823</span> <span style="color:#008000; ">101.34838521</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">class</span><span style="color:#308080; ">:</span> car<span style="color:#308080; ">,</span> score<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.83</span>
box coordinate x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008000; ">144.69975233</span> <span style="color:#008000; ">318.83762777</span> <span style="color:#008000; ">166.34090245</span>  <span style="color:#008000; ">84.78889614</span><span style="color:#308080; ">]</span>
</pre>

<img src="test-image-10-results.webp" width = "1000"/>


5. Analysis



Although the YOLO-v3 is considered as a state of the art deep learning object detection model, its performance is far from perfect:



It failed to detected several objects of interest, which appear clearly to the human eye
This may be due to the lack of diversity in the training data for certain classes
It also confused some classes, such as the cats and dogs in the first test images.
This yet another evidence that there is still much room for improvement for deep learning to become a reliable and consistent solution.


## 6. Future Work

* We plan to explore the following related issues:

  * Get more insights of some of the possible reasons behind the false negatives
  * Test the model on an annotated dataset and assess its performance using quantitative metrics such as:
    * Accuracy
    * Precision-recall (PR) curve
    * ROC curve
    * Read the original YOLO-v3 reference paper to understand the details of YOLO-v3 [9].


## 7. References

1. Larry Xiaochus. (April 14, 2021). YOLOv3. https://github.com/xiaochus/YOLOv3 
2. Joseph Redmon, (April 14, 2021). Ali Farhadi. YOLOv3: An Incremental Improvement. https://arxiv.org/abs/1804.02767
3. Joseph Chet Redmon. (April 14, 2021). YOLO: Real-Time Object Detection. https://pjreddie.com/darknet/yolo/ 
4. Microsoft. (April 14, 2021). COCO dataset. https://cocodataset.org/#home 
5. Ayoosh Kathuria. (April 14, 2021). How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 1. https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/ 
6. Ayoosh Kathuria. (April 14, 2021). How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 2. https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/ 
7. Ayoosh Kathuria. (April 14, 2021). How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 3. https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/ 
8. Ayoosh Kathuria. (April 14, 2021). How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 4. https://blog.paperspace.com/how-to-implement-a9. yolo-v3-object-detector-from-scratch-in-pytorch-part-4/ 
9. Ayoosh Kathuria. (April 14, 2021). How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 5. https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-5/ 
10. Joseph Redmon, Ali Farhadi. (April 14, 2021). YOLOv3: An Incremental Improvement. https://pjreddie.com/media/files/papers/YOLOv3.pdf










