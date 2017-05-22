# re-size images

from scipy import misc
import os

# read in images from file
cwd = os.getcwd()
print("Converting train images...")
for filename in os.listdir("data/tango_images_part1"):
	if filename.endswith(".png"):
		im = misc.imread(os.path.abspath("data/tango_images_part1/"+filename))
		im = misc.imresize(im, (100,75))
		misc.imsave(os.path.abspath("data/tango_images_part1_small/small_"+filename), im)

print("Converting train images...")
for filename in os.listdir("data/tango_images_part2"):
	if filename.endswith(".png"):
		im = misc.imread(os.path.abspath("data/tango_images_part2/"+filename))
		im = misc.imresize(im, (100,75))
		misc.imsave(os.path.abspath("data/tango_images_part2_small/small_"+filename), im)
