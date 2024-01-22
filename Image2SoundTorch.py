import torch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import numpy as np

# 0 - Set debugging flags
__i2sDebug__: bool = True
__interactDebug__: bool = True
if not __i2sDebug__:
	__interactDebug__ = False

# 1 - Load image, convert to tensor
FileName: str = "Smiley"
ImagePath: str = './Images/' + FileName + '.png'
img: np.ndarray = mpimg.imread(fname=ImagePath)
imgTensor: torch.Tensor = torch.from_numpy(img)

# 2 - Get dimensions and plot
height: int = len(img)
width: int = len(img[0])
if __i2sDebug__:
	print("Image shape: ",
		  imgTensor.shape)
	print("Height: ",
		  height)
	print("Width: ",
		  width)
	plt.imshow(img)
	plt.show()
	if __interactDebug__:
		input("Image Graphed...")

# 3 - Flatten the RGB colors into a single dimension
# Flatten tensor on the color dimension with a sum
FlattenedColorData = 3 - imgTensor[..., :3].sum(dim=-1)

if __i2sDebug__:
	print("Shape of FlattenedColorData:",
		  FlattenedColorData.shape)
	plt.imshow(FlattenedColorData)
	plt.show()
	if __interactDebug__:
		input("Color Data Flattened...")

# 4 - Transpose data
FlattenedColorDataT = torch.fliplr(FlattenedColorData.transpose(0,
																1))

if __i2sDebug__:
	print("Shape of Transposed Data:",
		  FlattenedColorDataT.shape)
	plt.imshow(FlattenedColorDataT.numpy())
	plt.show()
	if __interactDebug__:
		input("Data Transposed...")

# Set warping constants
percentWidthReduction: float = 0.9
percentHeightReduction: float = 0.4
midpointH: float = height / 2
midpointW: float = width / 2

# 5 - Warp data on the domain to place it in the correct part of the spectrum
# Warping Step 1: Shift y-coordinates
y_coords = torch.arange(height).view(1,
									 -1)
offset = (((percentHeightReduction / 2) * height) * (y_coords - midpointH) / midpointH).to(torch.int64)
new_y_coords = y_coords - offset
new_y_coords.clamp_(0,
					height - 1)  # Ensure coordinates are within bounds
WarpedData = FlattenedColorDataT[:, new_y_coords[0]]

# Warping Step 2: Non-linear transformation
selection_index = (torch.pow((y_coords / height),
							 1.0 / 3.5) * height).to(torch.int64)
selection_index.clamp_(0,
					   height - 1)
WarpedData2 = WarpedData[:, selection_index[0]]

# Warping Step 3: Shift x-coordinates
x_coords = torch.arange(width).view(-1,
									1)
offset = (((percentWidthReduction / 2) * width) * (x_coords - midpointW) / midpointW).to(torch.int64)
new_x_coords = x_coords - offset
new_x_coords.clamp_(0,
					width - 1)
WarpedData3 = WarpedData2[new_x_coords[:, 0]]

# Warping Step 4: Another shift in y-coordinates
offset = (((0.05 / 2) * height) * (y_coords - midpointH) / midpointH).to(torch.int64)
new_y_coords = y_coords - offset
new_y_coords.clamp_(0,
					height - 1)
WarpedData4 = WarpedData3[:, new_y_coords[0]]

# Perform FFT using PyTorch
t = torch.arange(width).float()
s = torch.fft.ifft(WarpedData4,
				   width)

# Plot the real part of the FFT result
plt.plot(t.numpy(),
		 s.real.numpy(),
		 label='real')
plt.legend()
plt.show()

# Concatenate the FFT results
totalOutput = torch.empty(0)
for row in WarpedData4:
	totalOutput = torch.cat((totalOutput, torch.fft.ifft(row,
														 10000).real))

# Plot the total output
plt.figure(figsize=(100, 10))
plt.plot(totalOutput.numpy())
plt.show()
plt.figure(figsize=(100, 10))
output_cut = totalOutput[int(percentWidthReduction * width * 0.5 * 10000): int((
																					   width * 10000) - percentWidthReduction * width * 0.5 * 10000)]
plt.plot(output_cut.numpy())
plt.show()

# Plot the imaginary part of the FFT result
plt.plot(t.numpy(),
		 s.imag.resolve_neg().numpy(),
		 '--',
		 label='imaginary')

plt.legend()
plt.show()

# Audio file writing remains the same as it requires the data in NumPy format
samplerate = 384000
totalOutputCut_np = output_cut.numpy().astype(np.float32)  # Convert to NumPy array for saving
wf.write("Sounds/" + FileName + ".wav",
		 samplerate,
		 totalOutputCut_np)
