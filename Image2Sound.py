import math

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import numpy as np

# 0 - Set debugging flags
__i2sDebug__: bool = True
__interactDebug__: bool = True
if not __i2sDebug__:
	__interactDebug__ = False

# 1 - Load image
FileName: str = "Smiley"
ImagePath: str = './Images/' + FileName + '.png'
img: np.ndarray = mpimg.imread(fname=ImagePath)

# 2 - Get dimensions and plot
height: int = len(img)
width: int = len(img[0])
if __i2sDebug__:
	print("Image shape: ",
		  img.shape)
	print("Height: ",
		  height)
	print("Width: ",
		  width)
	plt.imshow(img)
	plt.show()
	if __interactDebug__:
		input("Image Graphed...")

# 3 - Flatten the RGB colors into a single dimension
# Create empty ndarray and sum over it, removing the color dimension
FlattenedColorData = np.zeros([height, width])
for i in range(height):
	for j in range(width):
		FlattenedColorData[i][j] = (3 - (img[i][j][0] + img[i][j][1] + img[i][j][2]))

if __i2sDebug__:
	print("Shape of FlattenedColorData:",
		  FlattenedColorData.shape)
	plt.imshow(FlattenedColorData)
	plt.show()
	if __interactDebug__:
		input("Color Data Flattened...")

# 4 - Transpose data
FlattenedColorDataT: np.ndarray = np.fliplr(FlattenedColorData.transpose())

if __i2sDebug__:
	print("Shape of Transposed Data:",
		  FlattenedColorDataT.shape)
	plt.imshow(FlattenedColorDataT)
	plt.show()
	if __interactDebug__:
		input("Data Transposed...")

# Set warping constants
widthMultiplier: float = 0.9
heightMultiplier: float = 0.4
midpointH: float = height * 0.5
midpointW: float = width / 2

# Create ndarrays for warped data
WarpedData: list[np.ndarray] = [np.zeros([width, height]),
								np.zeros([width, height]),
								np.zeros([width, height]),
								np.zeros([width, height])]

# 5 - Warp data on the domain to place it in the correct part of the spectrum
# several warping steps to make the image correctly fit into the spectrum
# vertical dilation of the image
for i in range(width):
	for j in range(height):
		# calculate the index the point of the image needs to be brought to
		# this is a very naive algorithm, without any sort of antialiasing
		selectionIndex: int = j - int(heightMultiplier * midpointH * ((j / midpointH) - 1))
		WarpedData[0][i][selectionIndex] = FlattenedColorDataT[i][j]

warpExponent: float = 1.0 / 3.5
for i in range(len(WarpedData[0])):
	for j in range(len(WarpedData[0][i])):
		selectionIndex: int = math.floor(height * (j / height) ** warpExponent)
		WarpedData[1][i][j] = WarpedData[0][i][selectionIndex]

# horizontal dilation of the image
for i in range(len(WarpedData[1])):
	WarpedData[2][i - int(widthMultiplier * (i - midpointW))] = WarpedData[1][i]

for i in range(len(WarpedData[2])):
	for j in range(len(WarpedData[3][i])):
		WarpedData[3][i][j - int(0.05 * (j - midpointH))] = WarpedData[2][i][j]

if __debug__:
	for i, data in enumerate(WarpedData):
		plt.imshow(data)
		plt.show()
		input(("Warp Stage: " +
			   str(i)))

t: np.ndarray = np.arange(width)
s: np.ndarray = np.fft.ifft(a=WarpedData[3],
							n=width)
plt.plot(t,
		 s.real,
		 label='real')

totalOutput: np.ndarray = np.empty(0)

for row in WarpedData[3]:
	totalOutput = np.concatenate((totalOutput, np.fft.ifft(a=row,
														   n=10000))).real

plt.figure(figsize=(100, 10))
plt.plot(totalOutput)
plt.show()
plt.figure(figsize=(100, 10))
plt.plot(totalOutput[int(widthMultiplier * width * 0.5 * 10000): int((
																			 width * 10000) - widthMultiplier * width * 0.5 * 10000)])
plt.show()

totalOutputCut: np.ndarray = totalOutput[int(widthMultiplier * width * 0.5 * 10000): int((
																								 width * 10000) - widthMultiplier * width * 0.5 * 10000)]

plt.plot(t,
		 s.imag,
		 '--',
		 label='imaginary')

samplerate: int = 384000
fs: int = 100

t = np.linspace(start=0.,
				stop=1.,
				num=samplerate)

wf.write("Sounds/" + FileName + ".wav",
		 samplerate,
		 totalOutputCut.astype(np.float32))
