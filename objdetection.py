#!/usr/bin/python
import Image

orig = Image.open("kande1.pnm")
crop = orig.crop((149,263,329,327))
crop.save("test.ppm")

R,G,B = zip(*crop.getdata())

print R[:10]
print G[:10]
print B[:10]

