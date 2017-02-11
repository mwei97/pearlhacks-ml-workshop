# It's a bird, it's a plane, it's an Intro to Machine Learning
## Pearl Hacks 2017 workshop materials

This workshop does a high level overview of some basic machine learning concepts, and does a quick walkthrough of an example for supervised classification between images of birds and planes (hence, the title).

The accompanying slides can be found here:

[Workshop slides](https://docs.google.com/presentation/d/1AUG3VCjR0dpea5s_XT5_E2it7llQWFmOngWC1fHcCbg/edit?usp=sharing)

The raw data used in this workshop can be downloaded here:

[Birds and planes dataset](https://www.dropbox.com/s/7uyul7yyqwlyqzb/pearlhacks-ml-workshop-data.zip?dl=0)

However, the extracted color histogram features are already included with the code under `color_histogram_features`. To reproduce these, run from the command line
```
python extract_features.py
```

To run the classification, run from the command line
```
python svm_classifier.py
```
