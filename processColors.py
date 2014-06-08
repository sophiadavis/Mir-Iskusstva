'''
Sophia Davis
6/7/2014
processColors.py

Uses ImageMagick to calculate metrics about the colors in images inside each 
    'movement/small' subdirectory.
Saves output to a csv file (specified by first command line argument).

ImageMagick commands used:
    -unique color count: http://www.imagemagick.org/script/escape.php
    -average color: http://www.imagemagick.org/Usage/quantize/#colors
    -histograms: http://www.imagemagick.org/Usage/files/#histogram
'''
import sys
import glob
import subprocess
import csv
import re

def main():
    if len(sys.argv) < 2:
		sys.stderr.write('Usage: python ' + sys.argv[0] + ' imageInfoDataFile.csv\n')
		sys.exit(1)
    else:
        print "\nProcessing colors..."
        directories = [r"Aivazovskii/small", r"CritRealism/small", r"Icons/small", r"Modernism/small", r"SocRealism/small"]
        images = {}
        for sub in directories:
            images[sub] = glob.glob(sub + r"/*")
    
        # Define ranges of bins for histogram 
        cutPts = range(0, 255, 255/10)[:-1] # 9 bins with 25 values, 10th bin has 30
    
        with open(sys.argv[1], 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["Movement", "File", "NumColors", "AvgR", "AvgG", "AvgB"] + 
                                ["Gray" + str(cutPt) for cutPt in cutPts] + 
                                ["Red" + str(cutPt) for cutPt in cutPts] + 
                                ["Green" + str(cutPt) for cutPt in cutPts] +
                                ["Blue" + str(cutPt) for cutPt in cutPts])
        
            for sub in directories:
                print "\n--Inside " + sub 
                mvmt = re.search(r"(.*)/small", sub).group(1)
                for image in images[sub]:
                
                    name = re.search(r"small/(.*)_small", image).group(1)
                    print "****" + name
                    colorAttrs = getColorAttrs(image)
            
                    attrs = [mvmt, name] + colorAttrs
                    writer.writerow(attrs)
        f.close()
        print "\nDone.\n"

# Calculates number of unique colors present, average color, 
### and binned histogram counts (Gray, Red, Green, Blue channels)
def getColorAttrs(image):

    # Overall color metrics
    numColors = subprocess.check_output(["identify", "-format", "%k", image])
    avgColor = subprocess.check_output(["convert", image, "-scale", "1x1\!", "-format", "%[fx:int(255*r+.5)],%[fx:int(255*g+.5)],%[fx:int(255*b+.5)]", "info:"])
    avgColorList = avgColor.split(',')

    #### Bin values of histograms
    ## gray = luminance
    grayHistRaw = subprocess.check_output(["convert", image, "-colorspace", "Gray", "-format", "%c", "histogram:info:"])

    ## Red, green, blue color channels
    # ImageMagick converts each separate channel into grayscale 
    redHistRaw = subprocess.check_output(["convert", image, "-channel", "R", "-separate", "-format", "%c", "histogram:info:"])            
    greenHistRaw = subprocess.check_output(["convert", image, "-channel", "G", "-separate", "-format", "%c", "histogram:info:"])
    blueHistRaw = subprocess.check_output(["convert", image, "-channel", "B", "-separate", "-format", "%c", "histogram:info:"])

    grayHist = grayHistRaw.split("\n")[:-1] # Last item is an empty string
    redHist = redHistRaw.split("\n")[:-1]
    greenHist = greenHistRaw.split("\n")[:-1]
    blueHist = blueHistRaw.split("\n")[:-1]

    ##### Bin pixel counts
    grayBins = binHistogram(grayHist)
    redBins = binHistogram(redHist)
    greenBins = binHistogram(greenHist)
    blueBins = binHistogram(blueHist)
    
    return [numColors] + avgColorList + grayBins + redBins + greenBins + blueBins
    
def binHistogram(hist):
    # Regex patterns for parsing histogram output
    grayVal = "gray\((.*)\)"
    pixelCt = "^\s*(\d+):"
    
    tempBins = [0]*11 # Start with 11 bins for ease of indexing
    for line in hist: 
        val = re.search(grayVal, line).group(1) # ImageMagick converts each channel into gray (hence searching for gray, regardless)
        ct = re.search(pixelCt, line).group(1)
        key = int(val)/25 # Find appropriate bin index
        tempBins[key] += int(ct)
    bins = tempBins[0:-2] + [tempBins[-2] + tempBins[-1]] # Combine last two bins (into one bin containing 30-values)
    return bins

if __name__ == "__main__":
    main()