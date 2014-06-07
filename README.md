# Mir Iskusstva  
A program to categorize Russian paintings by artistic movement, using neural networks and backward propagation of errors.  
Sophia Davis  
AI Final Project, Spring 2014 

------------
## Important Files:  
#####Data
Original images are contained in the following subfolders:  

* Aivazovskii  
* CritRealism  
* Icons  
* Modernism  
* Socialist Realism 

Each image as saved as painter_paintingName. 

Rescaled images (used to obtain color features of each image) are located in the corresponding '/small' subdirectory (i.e. Aivazovskii/small)

#####Data Processing
resize.py -- rescales images to 100x100px using Imagemagick  
processColors.py -- extracts information about colors present in reduced-sized images, saves results to csv file  
rescaleData.R -- script to standardize input vectors (from results of processColors.py) to the interval [-2, 2]


#####Training/Testing Neural Network:  
crossValidate.py -- uses k-fold cross validation to evaluate the performance of a neural network  
node.py -- classes/functions for network nodes  
data.py -- classes/functions for data  
testNetworkParams.py -- tests the performance of network on training data, given specified combinations of network parameters (α (learning rate), μ (momentum rate), number of iterations, and number of hidden nodes and layers). Saves results to csv file.   
trainNetwork.py -- contains 'trainNetwork' function to train neural network given parameters and training set. Returns trained network and information about change in MSE over time.  


#####Evaluate Performance:  
trainingSetResults.R -- script to graph MSE of network on training set vs number of iterations (given output of testNetworkParams.py)   
testSetResults.R -- script to analyse performance of trained network on test data (given output of crossValidate.py)  

#####Results:  
imageColorData.csv -- output of processColors.py  
imageColorData_Standardized.csv -- contents of imageColorData.csv after standardization to [-2, 2] (using rescaleData.R). **This file was used to obtain all network performance statistics.**   
results_final.csv -- output of crossValidate.py, using data from imageColorData_Standardized.csv, and k = 20 different training and test sets (each test set contained one painting from each movement). Each network contained one hidden layer with 24 nodes (average number of input and output nodes), and was trained for 1500 iterations with α = 1000/(1000 + numIterations), and μ = α/2. 

These files contain information about the performance on a training set of networks trained using various combinations of α, μ, and hidden node structure (MSE after each iteration) -- output of testNetworkParams.py:  

* 1000 iterations, logistic activation function  
  * trainingPerformanceLogistic.csv   
  * trainingPerformanceLogistic2.csv  
* 1000 iterations, hyperbolic tangent activation function  
  * trainingPerformanceTanh.csv  
  * trainingPerformanceTanh2.csv  

### To Run: 

```
python crossValidate.py imageColorData_Standardized.csv output.csv
```
crossValidate.py is currently set to separate data into just 2 pairs of training/test sets and train a network for 10 iterations (α = 1000/(1000 + numIterations), μ = α/2, 1 layer of 24 hidden nodes) -- so this will run to completion fairly quickly and produce fairly awful results.

## Methodology
##### Name
Obviously, the first step was to name my project. I decided on Mir Iskusstva, after the early 20th century Russian avant-garde artistic movement and magazine, Мир искусства (World of Art).

##### Data 
Next I needed to track down paintings. I chose to work with five different artistic movements. In order to make image recognition as easy as possible for my ANN, I chose categories that were as different from each other as possible, and I made each category as homogeneous as possible by choosing to use works from as few artists as possible.  

I ended up with 20 paintings representing each of the following styles:

1. Icons -- Andrei Rublev
2. seascapes by Ivan Aivazovskii -- not really a movement (probably fits into Romanticism), but Aivazovskii's seascapes are so distinctive
3. Critical Realism -- Surikov (10 paintings) and Repin (10 paintings) 
4. Modernism -- Kandinskii
5. Socialist Realism -- mostly Deyneka, but I had to include work from a couple other artists as well to get to 20 paintings...

I downloaded digital versions of all paintings either from the [Tretyakov Gallery's online collection](http://www.tretyakovgallery.ru/ru/collection/_show/categories/_id/42) or from [Gallerix](http://gallerix.ru/), a huge digital collection of paintings, "founded with the goal of popularizing the art of painting among the vast masses of the population, and satisfying of the desire of seeing beautiful things among people who don't have the opportunity to personally visit museums." Between these two sites I still hadn't turned up 20 Socialist Realist paintings, so I turned to Google Images.

##### Image Processing
All image processing was conducted using [Imagemagick command line tools](http://www.imagemagick.org/script/command-line-tools.php).

I started off by making all images the same size -- 100px by 100px, ignoring aspect ratio (see `resize.py`). Imagemagick provides several ways to do this, but I chose to 'scale' the pictures ("minify / magnify the image with pixel block averaging and pixel replication, respectively") -- because this was the only description that I felt like I semi-understood what it was doing.

Some pictures suffered more from this resizing than others. For example, Aivazovskii's Pushkin put on a little weight:  
![image](https://github.com/sophiadavis/Mir-Iskusstva/blob/master/Aivazovskii/PushkinNaBereguChernogoMoria.jpeg) 

vs 

![image](https://github.com/sophiadavis/Mir-Iskusstva/blob/master/Aivazovskii/small/PushkinNaBereguChernogoMoria_small.jpeg)

So did Repin's Tolstoy:  
![image](https://github.com/sophiadavis/Mir-Iskusstva/blob/master/CritRealism/Repin_LNTolstoiBosoy.jpeg)

vs

![image](https://github.com/sophiadavis/Mir-Iskusstva/blob/master/CritRealism/small/Repin_LNTolstoiBosoy_small.jpeg)

Oh well.

Next, I figured out how to calculate a few simple metrics that my neural network could use to gauge the color-features of each image (see `processColors.py`). I used the total number of unique colors, the average red, green, and blue value. I also incorporated information from a luminosity histogram, and histograms of the red, green, and blue channels of each image. The Imagemagick command gave me the total number of pixels in the image with each color value (for each channel), and which I further divided into 10 ranges of color value.  

These color metrics were calculated from the resized versions of the paintings, and are saved in imageColorData.csv.


##### Algorithms

I implemented a feed-forward neural network, using the basic backwards propagation of errors formula to update weights between nodes.

After doing some research, I also added a momentum term. The basic idea is:  
Let ν_i be the amount by which a weight will be updated at iteration i.  

ν_i = α * gradient + μ * ν_(i - 1)    

Thus, whenever a weight is updated, a fraction of the previous weight update is added. In regular backprob, weights are updated based on information from one example at a time. The momentum term serves to provide some continuity, by including information "learned" from previous examples.  

[This paper](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf) by some people at the University of Toronto and [this one](http://axon.cs.byu.edu/papers/IstookIJNS.pdf) by some guys at Brigham Young University were most helpful in my gaining a basic understanding of what momentum is.

Initial weights were generated randomly from [-0.05, 0.05]. The training set order was randomized. I used the logistic function as my activation function. I also played around with using the hyperbolic tangent function, because [this guy](http://apps.carleton.edu/campus/library/) said it could help it converge faster.

##### Standardization of Input
I began by running my neural network on a random set of training data from the imageColorData.csv file. My goal was to play around with α, μ, and the hidden node structure until I found which set of network parameters reduced MSE on the training set the most.

No matter what combination of parameters I chose, no matter how long I ran my network, I couldn't get MSE to fall below .16ish, which is really bad (especially for the training set!). I felt like these guys:  


 
 

## Results

##### Performance on Training SEts

##### Performance on Test Sets
 