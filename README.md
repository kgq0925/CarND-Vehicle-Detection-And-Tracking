## CarND Vehicle Detection And Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/training_example.png
[image2]: ./output_images/non_vehicle_example.png
[image3]: ./output_images/hog_example.png
[image4]: ./output_images/slide_window_example.png
[image5]: ./output_images/slide_example.png
[image6]: ./output_images/output.png

## Rubric Points

### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Load Training Data

I started by reading in all the `vehicle` and `non-vehicle` images. The code for this step is contained in the second code cell of the IPython notebook.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image1]
![alt text][image2]

#### 1. Extracted HOG Features

The code for this step is contained in the 4th code cell of the IPython notebook.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### 2. HOG Parameters

`pixels_per_cell`: It is chosen based on the scale of the features important to do the classification. A very small size would blow up the size of the feature vector and a very large one may not capture relevant information. 

`cells_per_block`: A large size makes local changes less significant while a smaller block size weights local changes more.

`orientations`: It sets the number of bins in the histogram of gradients. The authors of the HOG paper had recommended a value of 9 to capture gradients between 0 and 180 degrees in 20 degrees increments.

I tried various combinations of parameters and found `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` gave the best result.

#### 3. Training Model

To do this I have chosen Support Vector Machines (SVM) as the classification algorithm. The code for this step is contained in the 10th code cell of the IPython notebook.

I tried to find the best parameters. The result is `C: 0.08`, `penalty: l2` and `loss: hinge`.

### Sliding Window Search

#### 1. Sliding Window Search

Try different windows size to test the detection accuracy. I finally decided to search window positions at the `window size: (96, 96)`, `overlap:(0.75, 0.75)`, and `Y-space:from 400 to 600`. The code for this step is contained in the 11th and 12th code cell of the IPython notebook.

![alt text][image4]

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Final Video Output.
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Pipeline.

1. Create a function called `detect` to implement pipeline.
2. Create a heat map using the raw image.
3. Add "heat" within `hot_windows` where a positive detection is identified by the svm classifier.
4. Rejected areas affected by false positives.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result:

![alt text][image6]

---

### Discussion

In this project, I use HOG and special binnings for feature extraction, and Linear SVM classifier was used for object detection. The SVM classifier can dectect vehicle well. But it cost about half an hour to apple the classifier on the 'project_video'. I think the performance is not good. To increase the efficiency of computing, parallel computing method may be a choice. I will all explore the deep learning approach to get a better performence.
