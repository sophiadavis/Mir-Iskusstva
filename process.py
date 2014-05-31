'''
Sophia Davis
5/30/2014
process.py
This script will process images by calculating various metrics about the colors,
using Imagemagick commands.
Information from each image will be stored in a csv file.
'''

import glob
import subprocess
import csv
import re

def main():
    jpgs = glob.glob('*.jpg')
    print jpgs
    
    # define ranges of bins for histogram 
    cutPts = range(0, 255, 255/10) # last bin will have 5 more color values
    
    # regex patterns for parsing histogram output
    grayVal = "gray\((.*)\)"
    pixelCt = "^.*(\d+):"
    
    with open('socialistRealist.csv', 'wb') as f: # use 'a' if this takes forever and we have problems 
        writer = csv.writer(f)
        writer.writerow(["File", "NumColors", "AvgColor"] + ["Gray" + str(cutPt) for cutPt in cutPts[:-1]])
        
        for jpg in jpgs:
        
            # overall color metrics
            numColors = subprocess.check_output(["identify", "-format", "%k", jpg])
            avgColor = subprocess.check_output(["convert", jpg, "-scale", "1x1\!", "-format", "%[pixel:p{1,1}]", "info:"])
            
            # bin values of grayscale histogram
            grayHistRaw = subprocess.check_output(["convert", jpg, "-colorspace", "Gray", "-format", "%c", "histogram:info:"])
            grayHist = grayHistRaw.split("\n")[:-1] # last item is an empty string
            
            tempBins = [0]*11 # 9 bins w 25 values, 10th with 30 values (11 bins for now -- simplifies indexing)

            for line in grayHist: 
                val = re.search(grayVal, line).group(1)
                ct = re.search(pixelCt, line).group(1)
                print line
                print int(val), int(ct)
                key = int(val)/25 # find appropriate bin
                print "adding to bin number" + str(key)
                tempBins[key] += int(ct)
            
            # combine last two bins
            bins = tempBins[0:-2] + [tempBins[-2] + tempBins[-1]]
            print bins
            
            attrs = [jpg, numColors, avgColor] + bins
            writer.writerow(attrs)
    f.close()

if __name__ == "__main__":
    main()