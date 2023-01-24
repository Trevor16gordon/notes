# Object Detection


When starting to talk about abject detection, before getting into convolutional neural networks and more, we first need to talk about how images are represented


## Image Processing Basics

**Image Representation**
- Represented in a 3d array (x by y by pixel color dim)
- Intensity represented from 0 to 255

**Image Gradient Vecotr**
- Contains the magnitude and direction of greatest increase for each pixel (each color)
- For each color, it contains a partial gradient in the x and the y direction
- Using both you can get the magnitude and the direction
- Can be calculated manually by looking at adjacent pixels
- Can be calculated by convolving a specific across the whole image
- Looking at the image gradient is useful for edge detection!


**Histogram of oriented Gradients**
- Binning of gradient magnitude and direction to acheive some useful derivied information
- Come back to this
- "HOG algorithm summarizes the information across all the gradient vectors in one image"


**Image Segmentation (Felzenszwalb’s Algorithm)**
- Useful to divide image into the smaller segments
- Can perform this using a graph structure where
    - Pixels are the nodes
    - Connections to other pixels based on the similarities of pixels (coordinates, colors, intensity)
    - Graph can be setup using grid coordinates (x,y position) / connected to 8 adjacent cells
    - Graph can also be setup using  nearest neighbour graph. Coordinates using feature space (x, y, r, g, b)
- Algorithm:
    - Start with each pixel belonging to it's own component
    - Calculate all edges. Sort them by highest edges (most similar pixels first)
    - For element in loop
        - If similarily is less than threshhold and pixels in different groups, merge pixels into the same group


**Selective Search**
- Another algorithm for image segmentation that will group existing region together based on similarity
- Calculated similarity scores of entire regions
- Doesn't work by just considering individual pixels


## Image Processing Using Neural Networks

**Convolutional Kernel**
- Examples of how a 2D convolutional kernel can work

**Alex Net 2012**
- 5 Layers

**VGG (Simonyan and Zisserman, 2014)**
- 19 layers and was very deep at the time
- More smaller convolutions. (Only 3x3). Less parameters so you can go deeper

**ResNet 2015**
- 152 Layers
- Residual "skip connections" which help with training such a deep network

## Evaluating Object Detection Models

**Mean Average Precision (maP)**
- Draw the precision recall curve and get the area underneith it (for each class)
- Average over the classes
- A detection is a true positive if it has "intersection over union” (IoU)" above some threshold. Usually 0.5


**Deformable Parts Model**
- Measure of quality of detection an object
- It contains 3 major score parts
    - A coarse root filter with a size that approximately covers an entire object
    - Multiple parts filters that a roughly half the size of the root filter.
    - A spatial model for detecting the relative position of the parts to the root
- When performing this you can get likely positions within a picture of some target.
- Author claims this can be represented by a CNN


**Overfeat**
- Integrates the object detection, localization and classification tasks all into one CNN
- Contains training a classifier and a localization regressor
- Classifier then connects to regressor to predict 4 locations of bounding box


## Region Based CNN


**R-CNN 2014**
- Using selective search, find a rough number of regions to select for classification using CNNs
- Workflow
    - Pre train CNNs on image recognition tasks
    - Using selective search break image down into regions
    - Select regions for object classification
    - Regions warped to have a fixed size.
    - Forward pass through CNN generates a feature vector, trained for each class independently 
    - Positive samples are those with IoU greater than 0.3
    - Localization errors corrected with a regression model


**Bounding Box Regression**
- Takes in bounding box predicted coordinates and the true bounding box coordinates
- Learns scale invariant translations between the centers
- Learns log-scale translations between the widths and heights


## Common Tricks

**Non Maximum Supression**
- Often models will find multiple bounding boxes for the same object
- After all bounding boxes discovered, sort by confidence score, remove the least confident ones.

**Hard Negative Mining**
- Classifier will find some weird texture areas which are not objects
- Add those to training with label no object to improve training


Everything discussed so far is quite slow as there are many separate steps


## Fast RNN
- Girshick (2015) improved the training procedure by unifying three independent models into one jointly trained framework and increasing shared computation results, named Fast R-CNN
- Workflow
    - Train classification CNNs
    - Using selective search for proposed regions
    - Alter pretrained CNN
        - Replace last max pooling layer with ROI pooling layer. ROI pooling layer outputs fixed length feature vectors of region proposals.
        - Replace last fully connected layer and last softmax layer(K classes) with the same thing over K+1 classes.
    - Model outputs two main things
        - Softmax estimator of K+1 classes per each ROI. Extra class for background
        - Bounding box regressor which predicts the offset from the ROI grid to the actual bounding box for each class.

- Loss includes loss from
    - True class label
    - Discrete probability distribution per each ROI
    - True bounding box
    - Predicted bounding box correction

**ROI Pooling**
- "type of max pooling to convert features in the projected region of the image of any size, h x w, into a small fixed window"

Fast RNN still slow because the region proposals still generated by a separate model


## Faster R-CNN 2016
- Integrates the region proposal into the same network
- Workflow
    - Pre train models on classification tasks
    - Train a RPN (Region Proposal Network)
        - Slide a small window across images
        - At center of regioin, predict multiple regions of various sizes
    - Train a Fast CNN using the proposals generated with RPN
    - Add them together.

- Loss includes more details from the window finding network



## Mask RNN
- Builds on Faster RNN by adding pixel level assignment to the output regions
- Trains a segmentation branch in parallel
-  mask R-CNN improves the RoI pooling layer (named “RoIAlign layer”) 


## SSD 
- A single-shot model. Everything including 
- Predict category score with a a fixed set of bounding boxes
- Predict on different scales and ratios.
- Includes a pre-trained classification network.
    - Replaces last layers with convolutions
    - Drops classification layer
- Includes extra feature layer networks
    - As we go deeper the resolution decreases. So they use all of the last layers to loook for different size objects.
    

## YOLO
- Encode the training label as a vector including 4 elements for boudning box and one hot encoding for the classes
- Works for single class classification
- For multiclass, split image into grid
- Only the cell with the center of the object is responsible for predicting the object
- Loss function includes a normalization so that error between large boxes isn't more expensive than error for smaller bounding boxes.
- Very fast: 30 FPS or 150 FPS
- Drawbacks: Has trouble if many objects in a single cell.


## Improving Yolo
- 
- Anchor boxes, Dimension priors


## Yolo V3 and above
-  More convolutional layers: 53 compared to __
- Includes residual blocks











