'''
Sophia Davis
5/30/2014
processColors.py

Calculates metrics about the colors in images, using Imagemagick
Stores output in a csv file
'''

import sys
import glob
import subprocess
import csv
import re

def main():
    if len(sys.argv) < 2:
		sys.stderr.write('Usage: python ' + sys.argv[0] + ' trainingDataFile.csv\n')
		sys.exit(1)
    else:
        print "\nProcessing colors..."
        directories = [r"Aivazovskii/small", r"CritRealism/small", r"Icons/small", r"Modernism/small", r"SocRealism/small"]
        images = {}
        for sub in directories:
            images[sub] = glob.glob(sub + r"/*")
    
        # define ranges of bins for histogram 
        cutPts = range(0, 255, 255/10)[:-1] # 9 bins with 25 values, 10th bin has 30
    
        with open(sys.argv[1], 'wb') as f: # use 'a' if this takes forever and we have problems 
            writer = csv.writer(f)
            writer.writerow(["Movement", "File", "NumColors", "AvgR", "AvgG", "AvgB"] + 
                                ["Gray" + str(cutPt) for cutPt in cutPts] + 
                                ["Red" + str(cutPt) for cutPt in cutPts] + 
                                ["Green" + str(cutPt) for cutPt in cutPts] +
                                ["Blue" + str(cutPt) for cutPt in cutPts])
        
            for sub in directories:
                print "\n--Inside " + sub 
                mvmt = re.search(r"(.*)/small", sub).group(1)
                for image in images[sub][0:5]:
                
                    name = re.search(r"small/(.*)_small", image).group(1)
                    print "****" + name
                
                    # overall color metrics
                    numColors = subprocess.check_output(["identify", "-format", "%k", image])
                    avgColor = subprocess.check_output(["convert", image, "-scale", "1x1\!", "-format", "%[fx:int(255*r+.5)],%[fx:int(255*g+.5)],%[fx:int(255*b+.5)]", "info:"])
                    avgColorList = avgColor.split(',')
            
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
            
                    attrs = [mvmt, name, numColors] + avgColorList + grayBins + redBins + greenBins + blueBins
                    writer.writerow(attrs)
        f.close()
        print "\nDone.\n"
    
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