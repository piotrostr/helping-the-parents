# helping-the-parents

This project's aim is to create an easy-to-use eye-tracking software based on the paper:

```
@inproceedings{Park2020ECCV,
  author    = {Seonwook Park and Emre Aksan and Xucong Zhang and Otmar Hilliges},
  title     = {Towards End-to-end Video-based Eye-Tracking},
  year      = {2020},
  booktitle = {European Conference on Computer Vision (ECCV)}
}
```

The main goals of the project are to deliver an efficient pipeline for camera calibration (both intrinsic and extrinsic) as well as real-time eye patch extraction.

### Set-up

```pip install -r requirements.txt``` but be sure to ```git-lfs pull``` as well to get the sample data and models pulled

Place the calibration images with chessboards from a variety of angles into ```./data```. Ideally it should be around 10 photos, with all the squares of the chessboard visible among all the images. Then place ```scene.mp4``` capture of screen and ```face.mp4``` capture of webcam into ```./data```. 

The angle at which the webcam is recording has to be the same as for the images being calibrated!

### Preprocessing steps:

1) intrinsic matrix calibration using opencv and ChArUco board [5]
2) extrinsic camera calibration using mirrors [1] !
3) undistort the frames
4) detect face (skip this as blazeface does it well)
5) detect face-region landmarks (use face_alignment and blazeface)
6) perform 3D morphable model (3DMM) to 3D landmarks [2] !
7) apply 'data normalization' for yielding eye patches [3, 4] under assumptions:
    - virtual camera is located 60cm away from the gaze origin
    - focal length of ~1800mm

[1] https://www.jstage.jst.go.jp/article/ipsjtcva/8/0/8_11/_pdf/-char/en

[2] https://openresearch.surrey.ac.uk/discovery/delivery/44SUR_INST:ResearchRepository/12139198320002346#13140605970002346

[3] https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sugano_Learning-by-Synthesis_for_Appearance-based_2014_CVPR_paper.pdf

[4] https://www.perceptualui.org/publications/zhang18_etra.pdf

[5] https://docs.opencv.org/4.5.2/dc/dbb/tutorial_py_calibration.html

### Data normalization

The algorithm for data normalization comes from

```
@inproceedings{10.1145/3204493.3204548,
  author    = {Zhang, Xucong and Sugano, Yusuke and Bulling, Andreas},
  title     = {Revisiting Data Normalization for Appearance-Based Gaze Estimation},
  year      = {2018},
  url       = {https://doi.org/10.1145/3204493.3204548},
  booktitle = {Proceedings of the 2018 ACM Symposium on Eye Tracking Research &amp; Applications},
}
```

and can be found in the ```normalize_data.py``` file.
