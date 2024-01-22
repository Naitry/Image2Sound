# Image2Sound

Initiallly created January 2023 when the Fourier transform and this spectrogram
app https://musiclab.chromeexperiments.com/spectrogram/ were mentioned in class and I wanted a very visceral
demonstration of the math.

Simple python notebook to encode images in sound files using an inverse fourier transform

I am planning on formalizing it into a more robust script when I have the time

Disclaimer::
This was designed specifically for use with https://musiclab.chromeexperiments.com/spectrogram/ and thus will likely not
display with proper proportions out of the box on other spectrograms.

### venv setup commands (linux)

this is at least how I would set up to use the project

```angular2html
python -m venv venv
source ./venv/bin/activate
pip install torch matplotlib scipy numpy
pip install --upgrade pip
```