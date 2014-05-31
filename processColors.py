'''
Sophia Davis
5/30/2014
processColors.py

Calculates metrics about the colors in images, using Imagemagick
Stores output in a csv file
'''

import glob
import subprocess
import csv
import re

def main():
    directories = [r"Aivazovskii/small", r"CritRealism/small", r"Icons/small", r"Modernism/small", r"SocRealism/small"]
    images = {}
    for sub in directories:
        images[sub] = glob.glob(sub + r"/*")
    print images
    
    # define ranges of bins for histogram 
    cutPts = range(0, 255, 255/10)[:-1] # 9 bins with 25 values, 10th bin has 30
    
    with open('imageColorData.csv', 'wb') as f: # use 'a' if this takes forever and we have problems 
        writer = csv.writer(f)
        writer.writerow(["Movement", "File", "NumColors", "AvgColor"] + 
                            ["Gray" + str(cutPt) for cutPt in cutPts] + 
                            ["Red" + str(cutPt) for cutPt in cutPts] + 
                            ["Green" + str(cutPt) for cutPt in cutPts] +
                            ["Blue" + str(cutPt) for cutPt in cutPts])
        
        for sub in directories:
            for image in images[sub]:
                # overall color metrics
                numColors = subprocess.check_output(["identify", "-format", "%k", image])
                avgColor = subprocess.check_output(["convert", image, "-scale", "1x1\!", "-format", "%[fx:int(255*r+.5)],%[fx:int(255*g+.5)],%[fx:int(255*b+.5)]", "info:"])
            
                #### bin values of histograms
                ## gray = luminance
                grayHistRaw = subprocess.check_output(["convert", image, "-colorspace", "Gray", "-format", "%c", "histogram:info:"])
            
                ## red, green, blue color channels
                # imageMagick converts each separate channel into grayscale 
                redHistRaw = subprocess.check_output(["convert", image, "-channel", "R", "-separate", "-format", "%c", "histogram:info:"])            
                greenHistRaw = subprocess.check_output(["convert", image, "-channel", "G", "-separate", "-format", "%c", "histogram:info:"])
                blueHistRaw = subprocess.check_output(["convert", image, "-channel", "B", "-separate", "-format", "%c", "histogram:info:"])
            
                grayHist = grayHistRaw.split("\n")[:-1] # last item is an empty string
                redHist = redHistRaw.split("\n")[:-1]
                greenHist = greenHistRaw.split("\n")[:-1]
                blueHist = blueHistRaw.split("\n")[:-1]
            
                ##### bin pixel counts
                grayBins = binHistogram(grayHist)
                redBins = binHistogram(redHist)
                greenBins = binHistogram(greenHist)
                blueBins = binHistogram(blueHist)
            
                print "Gray Histogram Values:" + str(grayBins)
                print "Red Histogram Values:" + str(redBins)
                print "Green Histogram Values:" + str(greenBins)
                print "Blue Histogram Values:" + str(blueBins)
                attrs = [sub, image, numColors, avgColor] + grayBins + redBins + greenBins + blueBins
                print attrs
                writer.writerow(attrs)
    f.close()
    
def binHistogram(hist):
    # regex patterns for parsing histogram output
    grayVal = "gray\((.*)\)"
    pixelCt = "^\s*(\d+):"
    
    tempBins = [0]*11 # start with 11 bins for ease of indexing
    for line in hist: 
        val = re.search(grayVal, line).group(1) # imageMagick converts each channel into gray (hence searching for gray, regardless)
        ct = re.search(pixelCt, line).group(1)
        key = int(val)/25 # find appropriate bin index
        tempBins[key] += int(ct)
    bins = tempBins[0:-2] + [tempBins[-2] + tempBins[-1]] # combine last two bins (into one bin containing 30-values)
    return bins

if __name__ == "__main__":
    main()