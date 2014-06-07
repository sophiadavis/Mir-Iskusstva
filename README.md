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
![big Pushkin](/Aivazovskii/PushkinNaBereguChernogoMoria.jpeg) ![small, fat Pushkin](/Aivazovskii/small/PushkinNaBereguChernogoMoria_small.jpg)

So did Repin's Tolstoy:  
![big Tolstoy](/CritRealism/Repin_LNTolstoiBosoy.jpeg) ![squat Tolstoy](/CritRealism/small/Repin_LNTolstoiBosoy_small.jpg)

Oh well.

Next, I figured out how to calculate a few simple metrics that my neural network could use to gauge the color-features of each image (see `processColors.py`). I used the total number of unique colors, the average red, green, and blue value. I also incorporated information from a luminosity histogram, and histograms of the red, green, and blue channels of each image. The Imagemagick command gave me the total number of pixels in the image with each color value (for each channel), and which I further divided into 10 ranges of color value.  

These color metrics were calculated from the resized versions of the paintings, and are saved in imageColorData.csv.


##### Algorithms

I implemented a feed-forward neural network, using the basic backwards propagation of errors formula to update weights between nodes. There were 44 input nodes (one for each of the color features) and 5 output nodes (one for each possible classification). Initial weights were generated randomly from [-0.05, 0.05]. The order of the training set was randomized, and the network was trained on each painting in turn.  
Although my data was categorical (each item belongs to 1 of 5 classifications), node output was still continuous in nature. Thus, error at the output nodes was calculated as:  
1.0 - (node output) -- if the painting currently being processed belonged to the category represented by the output node  
0.0 - (node output) -- otherwise. 

After doing some research, I also added a momentum term. The basic idea is:  
Let ν_i be the amount by which a weight will be updated at iteration i.  

ν_i = α * gradient + μ * ν_(i - 1)    

Thus, whenever a weight is updated, a fraction of the previous weight update is added. In regular backprob, weights are updated based on information from one example at a time. The momentum term serves to provide some continuity, by including information "learned" from previous examples.  

[This paper](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf) by some people at the University of Toronto and [this one](http://axon.cs.byu.edu/papers/IstookIJNS.pdf) by some guys at Brigham Young University were most helpful in my gaining a basic understanding of what momentum is.

I used the logistic function as my activation function. I also played around with using the hyperbolic tangent function, because [this guy](http://apps.carleton.edu/campus/library/) said it could help it converge faster.

##### Standardization of Input
I began by running my neural network on a random set of training data from the imageColorData.csv file. My goal was to play around with α, μ, and the hidden node structure until I found which set of network parameters reduced MSE on the training set the most.

No matter what combination of parameters I chose, no matter how long I ran my network, I couldn't get MSE to fall below 0.16, which is really bad (especially for the training set!). I felt like these guys:  
![Burlaki hate ANNs](/ReadmeImages/Repin_BurlakiNaVolge_two.jpg)

I finally realized that the hidden nodes were always outputting 1.0. Of course MSE wasn't decreasing: MSE is calculated at the output nodes, and the input to the output nodes was always the same. Because my input values (mostly in numbers of pixels) were so enormous, no matter how small I made my initial weights, after a few iterations, the activation function at the hidden nodes would never return anything but 1.0. 

I did some more research, and eventually found [this paper](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/), by an ANN God named Warren S. Sarle. It has a whole section on why and how inputs to neural networks should be standardized (search "Subject: Should I normalize/standardize/rescale"). I used the formulae given by Sarle to standardize all of my input vectors to lie in the range [-2, 2] (see `rescaleData.R`, with standardized data set saved as imageColorData_Standardized.csv).

As soon as I started training a network using the standardized data set, MSE started decreasing right away. Lovely. I felt like this:  
![Рабочий и Колхозница](/ReadmeImages/success.jpg)  
 
## Results

##### Performance on Training Sets
Once my neural network appeared to be capable of learning, I wrote a script that would train neural networks using several different combinations of parameters, and save the MSE from each iteration in a csv file (see `testNetworkParams.py`).

I couldn't get anything to successfully reduce MSE when using a hyperbolic tangent function. Perhaps if I spent more time messing with α and μ, I could get something decent. However, since some of the networks I trained using a logistic activation had shown some evidence of convergence, I didn't press the issue. 
![TanH](/ReadmeImages/TanHTrainingResults.png)

Using a logistic function was, in general, more successful. Of the following parameter combinations (including some repeats), numbers 8 through 13 produced networks with fairly low MSE after 1000 iterations, and are represented in the graph below.

|Model|α|μ|Hidden Structure|Final Avg. Weight Change|  
| :------------- |:-------------:|:-------------:| :-------------:|:-------------:|  
|1|0.4|0.9|[24]|4.816977e-09|  
|2|1000/(1000 + x)|0.9|[24]|2.861154e-08|  
|3|1000/(1000 + x)|0.9|[32, 24]|2.696963e-09|  
|4|1000/(1000 + x)|0.9|[32]|1.446811e-10|  
|5|1000/(1000 + x)|0.9|[44]|6.896931e-14|  
|6|1000/(1000 + x)|0.9|[5]|4.425181e-04|  
|7|1000/(1000 + x)|1.0 - 3.0/(x + 5.0)|[24]|3.121103e-10|  
|**8|1000/(1000 + x)|α/2|[24]|3.850372e-06|**  
|**9|1000/(1000 + x)|α/2|[24]|1.503116e-04|**  
|10|1000/(1000 + x)|α/2|[32]|1.119652e-06|  
|**11|1000/(1000 + x)|α/2|[44]|9.061062e-06|**  
|12|1000/(1000 + x)|α/2|[5]|2.555765e-03|  
|13|1000/(1000 + x)|α/2|[24, 32]|1.172053e-03|  
|14|1000/(2 * (1000 + x))|0.9|[24]|7.026136e-04|  
|15|2 * μ|1000/(2*(1000 + x))|[24]|7.701949e-04|  

![Logistic](/ReadmeImages/LogTrainingResults.png)

Models 8, 9, and 11 (bolded) performed best. 8 and 9 were actually the same combination of parameters. Model 11 also had the same α and μ -- but whereas 8 and 9 had a single layer of 24 hidden nodes (the average of the number of input and output nodes), 11 had 44 hidden nodes (the same number as the number of input nodes). To avoid overfitting the data, I opted for the parameters used in models 8 and 9. These also had relatively high final average weight changes -- so perhaps the weights would change even more (and the network would show even better convergence) if I could train the network for longer.


##### Performance on Test Sets
After settling on the logistic activation function, α = 1000/(1000 + x), μ = α/2, and a single layer of 24 hidden nodes, I proceeded to cross validate the neural network.

I divided the data into 20 unique training/test set pairs. Each test set contained one picture representing each category, and the corresponding training set contained all other pictures. The network was trained for 1500 iterations. (see `crossValidate.py`)

Now, each output node returned a value in a continuous range. I used the classification corresponding to the output node with the maximum output value as the network's "prediction" for a given painting. Using that metric, there was a 55% overall success rate. The table below lists the number of successes by trial, as well as the network's MSE on both training and test sets.

|Test Number|Correct|Incorrect|TrainingSetMSE|TestSetMSE|  
| :-------------|:-------------:|:-------------:| :-------------:|-------------:|   
|1|3|2|0.021113856|0.18510327|  
|**2**|**5**|**0**|**0.019018235**|**0.03510553**|  
|3|3|2|0.031612875|0.11206679|  
|4|4|1|0.019894575|0.04048364|  
|5|2|3|0.016983443|0.22665282|  
|6|3|2|0.029581011|0.11527369|  
|7|2|3|0.011165801|0.16166198|  
|8|2|3|0.020525809|0.19985612|  
|9|2|3|0.006390899|0.19643967|  
|10|3|2|0.012834621|0.12223814|  
|11|3|2|0.012815334|0.17885093|  
|12|2|3|0.006406209|0.19219258|  
|13|3|2|0.008480505|0.15246886|  
|14|2|3|0.008598175|0.28520785|  
|15|1|4|0.014824077|0.26372471|  
|16|3|2|0.033807571|0.09967330|  
|17|4|1|0.010711561|0.05955265|  
|18|2|3|0.025845815|0.19930991|  
|19|4|1|0.033968625|0.04044056|  
|20|2|3|0.021225349|0.21411173|  

However, by considering only the classification from the output node with the highest returned value, we're losing a lot of information. The following barcharts show the average prediction of each output node, for all paintings by category.

![Logistic](/ReadmeImages/Aivazovskii_Results.png)  
![Logistic](/ReadmeImages/CritRealism_Results.png)  
![Logistic](/ReadmeImages/Icons_Results.png)  
![Logistic](/ReadmeImages/Modernism_Results.png)  
![Logistic](/ReadmeImages/SocRealism_Results.png)  

In conclusion, this is awesome! Победа!
