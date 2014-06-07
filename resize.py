'''
Sophia Davis
5/31/2014
resize.py

Resizes images to 100x100 px (ignoring original aspect ratio) and
    saves new images into 'small' subdirectory.
Uses ImageMagick's 'scale' operator: 
    "minify / magnify the image with pixel block averaging and pixel replication, respectively"
    http://www.imagemagick.org/script/command-line-options.php#scale
'''
import glob
import os
import re

def main():
    print "\nResizing images..."
    directories = ["Aivazovskii", "CritRealism", "Icons", "Modernism", "SocRealism"]
    images = {}
    
    # Get list of all images (.jpg or .jpeg)
    for sub in directories:
        images[sub] = glob.glob(sub + r"/*g")
    
    # Resize each image, rename, and save to "small" sub-folder
    for sub in directories:
        print "\n--Inside " + sub
        for image in images[sub]:
            name = re.search(r"/(.*)[.]", image).group(1) # Extract 'artist_title' from file name
            print "****" + name
            
            command = "convert -scale 100x100! " + image + ' ' + sub + "/small/" + name + "_small.jpg"
            os.system(command)
            
    print "\nDone.\n"
if __name__ == "__main__":
    main()