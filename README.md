# Kalman-tracking
This code contains a standalone version of the tracking code included in DELPHES.
The calculation of the track covariance matrix can be done either with standard full covariance matrix of the measurements, including multiple scattering effects, or with a Kalman filter approach.

## Standalone C script (ROOT)

Running instructions:

```
root

root> .L examples/LoadTrk.c+

root> LoadTrk()

root> KalmanCheck("./Geometries/GeoIDEA_NewDCH.txt")
```

This program will generate plots using both the standard version or the Kalman version of the code.

```
root> PlotGeo("./Geometries/GeoIDEA_NewDCH.txt")
```

will make a simple plot of the geometry and display the material as a function of the polar angle or its cosine.

## Python multi-detector comparison script

The `perf_multi.py` script allows to compare tracking performance across multiple detector geometries. It produces resolution scans, geometry plots, material budget plots, and optionally a LaTeX beamer report.

### Environment setup (lxplus)

To run on lxplus you need to set up the LCG software stack:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
export PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux:$PATH
```

The second export is needed to produce the LaTeX report.

### Example

```bash
python3 perf_multi.py \
  --card Geometries/GeoIDEA_NewDCH.txt --label IDEA --bfield 2.0 --doKalman true --doMS true --doRes true \
  --card Geometries/GeoCLD.txt --label CLD --bfield 2.0 --doKalman true --doMS true --doRes true \
  --inc examples/classes --compile --workers 60 --latex --npoints 100 \
  --outdir .
```

When using the `--latex` flag, a beamer PDF report like [this one](https://mselvaggi.web.cern.ch/maps/report_idea-MSon-ResOn-KalmanOff-2T_cld-MSon-ResOn-KalmanOff-2T/report_idea-MSon-ResOn-KalmanOff-2T_cld-MSon-ResOn-KalmanOff-2T.pdf) will be produced.
