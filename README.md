# WaveNet-based approximation of a normal-hearing cochlear filter and hair-cell transduction model
This repository contains code for the WaveNet-based approximation of the normal-hearing cochlear filtering and hair-cell transduction stages of the Zilany &amp; Bruce auditory model (J. Acoust. Soc. Am. 120(3), 1446–1466, 2006).

When using the WaveNet auditory model, please cite the following paper (see BibTex information at the bottom of this page):

**Nagathil, A., & Bruce, I. C. (2023). WaveNet-based approximation of a cochlear filtering and hair cell transduction model. The Journal of the Acoustical Society of America, 154(1), 191-202.**
https://doi.org/10.1121/10.0020068

### Important comment

After an initial release of the model parameters, the model was re-trained using the same model configuration as described in the paper. However, instead of halving the learning rate after stagnation of the validation loss, here we set the new
learning rate to a value of 0.8 times the previous learning rate. This improved the overall approximation accuracy, especially at higher characteristic frequencies and lower sound pressure levels.
Please indicate this change when citing the paper.

## Instructions for use

This repository contains code written in Python/PyTorch and MATLAB and allows a comparison of the WaveNet approximation and the original auditory model. To run the WaveNet model only,
open a command line window and type in the following command:

```
python3 run_model.py --file audio_file --spl stimdb --totdur stimT --segdur seg_dur --proc proc_unit
```
If the input arguments are discarded, default parameters are used.

| Argument  | Description  | Default |
|----------|:-------------:|------:|
| audio_file | Path to audio file | './audio/61-70968-0000.flac' |
| stimdb | Sound pressure level (in dB). | 60 |
| stimT | Total duration of signal to be processed, starting at the beginning (in s). | 4 |
| seg_dur | Duration of non-overlapping segments, into which the signal is split (in s). | 1 |
| proc_unit | Processing unit (cpu or cuda). | "cpu" |

For a comparison of the original auditory model (written in MATLAB/C) and the WaveNet-based approximation, run the script
```
compare_model.m
```
in MATLAB. It executes both model implementations, measures computation times, plots IHCograms, and measures the approximation quality in terms of a signal-to-distortion ratio (SDR) as specified in Eq. (3) in the Nagathil & Bruce (2023) paper. Note, that the MATLAB implementation is executed on all available CPU cores in parallel. The source of the original model code can be found here: https://www.ece.mcmaster.ca/~ibruce/zbcANmodel/zbcANmodel.htm

## Prerequisites

For running the MATLAB/C code the Parallel Computing Toolbox is required. If this toolbox is not available, change the **parfor** statement in the function 'original_model/generate_ihcgram_BEZ2018_parallelized.m' to a **for** statement.

The Python/PyTorch code for the WaveNet model was tested under Ubuntu 22.04 in a dedicated Anaconda environment. After installing Anaconda on your system (https://www.anaconda.com), create
an environment in the command line

```
conda create --name wnTest python
```
and activate it:

```
conda activate wnTest
```
Install all required packages by using the **requirements.txt** file provided in the repository

```
pip install -r requirements.txt
```
or install them manually:

```
pip install torch numpy scipy pyyaml librosa
```
## Contact

For questions or comments, please contact the authors:

- Anil Nagathil: anil.nagathil@rub.de

- Ian C. Bruce: brucei@mcmaster.ca

## BibTex:
```
@article{nagathil2023,
  title={WaveNet-based approximation of a cochlear filtering and hair cell transduction model},
  author={Nagathil, Anil and Bruce, Ian C},
  journal={The Journal of the Acoustical Society of America},
  volume={154},
  number={1},
  pages={191--202},
  year={2023},
  publisher={AIP Publishing}
}
``` 
