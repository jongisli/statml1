#!/usr/bin/python
import Image

orig = Image.open("kande1.pnm")
crop = orig.crop((149,263,329,327))
crop.save("test.ppm")

for (R,G,B) in crop.getdata():
   print "Red: %d, Green: %d, Blue: %d" % (R,G,B)
