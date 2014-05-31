'''
Sophia Davis
5/31/2014
resize.py

Resizes images to 100x100 px (ignoring original aspect ratio) and
    saves new images into 'small' subdirectory
Uses ImageMagick's 'scale' operator: 
    "minify / magnify the image with pixel block averaging and pixel replication, respectively"
    http://www.imagemagick.org/script/command-line-options.php#scale
'''
import glob
import os
import re

def main():
    directories = ["Aivazovskii", "CritRealism", "Icons", "Modernism", "SocRealism"]
    images = {}
    for sub in directories:
        images[sub] = glob.glob(sub + r"/*g") # only take jpeg or jpg
    print images
    for sub in directories:
            for image in images[sub]:
                name = re.search(r"/(.*)[.]", image).group(1) # extract 'artist_name' from file name
                command = "convert -scale 100x100! " + image + ' ' + sub + "/small/" + name + "_small.jpg"
                os.system(command)

if __name__ == "__main__":
    main()