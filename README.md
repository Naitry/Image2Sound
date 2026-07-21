# Image2Sound

Image2Sound is an educational signal-processing project that encodes an image
into a WAV file using the inverse Fourier transform. When the generated audio
is viewed in a spectrogram, the source image becomes visible in the frequency
domain.

I began the project in January 2023 after Fourier transforms and the
[Chrome Music Lab Spectrogram](https://musiclab.chromeexperiments.com/spectrogram/)
were discussed in class. The goal was to create a direct, audible demonstration
of how frequency-domain data maps to a time-domain signal.

## Project status

This repository is a completed educational proof of concept, not a maintained
production tool.

- `Image2Sound.py` is the original working NumPy implementation used to produce
  the included example WAV files.
- `Image2SoundTorch.py` is an experimental PyTorch port that replaces many of
  the array and transform operations with tensor operations.
- `Image2Sound.ipynb` records the earlier notebook-based development process.

Both scripts currently use hard-coded input and output paths and display
interactive diagnostic plots. They are preserved to document the experiment
rather than provide a polished command-line application.

## How it works

The scripts:

1. Load an RGB image and reduce it to a two-dimensional intensity map.
2. Transpose and geometrically warp the image to fit the target spectrogram's
   time and frequency axes.
3. Treat each transformed row as frequency-domain magnitude data.
4. Apply an inverse Fourier transform to generate successive time-domain audio
   segments.
5. Concatenate the segments and write the result as a 384 kHz WAV file.

The warping constants were tuned specifically for the Chrome Music Lab
Spectrogram. Other spectrogram tools may display the image with different
proportions.

## Repository contents

- `Images/`: source images and an intermediate warped example
- `Sounds/`: generated WAV examples
- `Image2Sound.py`: NumPy implementation
- `Image2SoundTorch.py`: experimental PyTorch implementation
- `Image2Sound.ipynb`: development notebook

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib numpy scipy torch
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Running the prototype

Set `FileName` near the top of either script to the name of a PNG file in
`Images/`, then run:

```bash
python Image2Sound.py
```

The script displays several intermediate plots and pauses for confirmation
between stages. The generated audio is written to `Sounds/<FileName>.wav`.

## Known limitations

- Input and output names are configured in the source rather than through
  command-line arguments.
- Image warping uses hand-tuned nearest-neighbor index transformations without
  antialiasing.
- The output geometry is calibrated for one spectrogram viewer.
- The scripts prioritize visualizing the transform process over runtime or
  memory efficiency.
