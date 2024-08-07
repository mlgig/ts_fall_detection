# Accurate and Efficient Real-World Fall Detection Using Time Series Techniques

## Abstract

Falls pose a significant health risk, particularly for older peo-
ple and those with specific medical conditions. Therefore, timely fall de-
tection is crucial for preventing fall-related complications. Existing fall
detection methods often have limitations, such as high false alarm rates
or reliance on handcrafted features. Additionally, most approaches to
fall detection are evaluated on simulated falls. This paper explores a new
fall detection approach leveraging real-world fall data and state-of-the-
art time series techniques. The proposed method eliminates the need for
manual feature engineering and has efficient training times. We evaluate
our approach on the FARSEEING dataset, a large collection of real-
world falls, and two large simulated falls datasets, FallAllD and SisFall.
Our approach achieves high accuracy (F1 score up to 97.2%). Further-
more, we investigate the performance of models trained on simulated
data and tested on real-world data. This research presents a real-time
fall detection framework with potential for real-world implementation.

## Link to full paper
You can read the full paper [here](https://www.researchgate.net/publication/382940726_Accurate_and_Efficient_Real-World_Fall_Detection_Using_Time_Series_Techniques) or download [here](paper/Accurate%20and%20Efficient%20Fall%20Detection%20Using%20Time%20Series%20Techniques.pdf). All the results can be seen in this repository, summarised in [evaluation.ipynb](evaluation.ipynb).

## Using the Code

To rerun the code, the preprocessed versions of `FallAlld` and `SisFall` used in this study can be downloaded [here](https://drive.google.com/file/d/1ysbgiGd0jDDtkTGu2HYJGiSEUawLJt8b/view?usp=sharing). The FARSEEING dataset can be requested for [here](https://farseeingresearch.eu/the-farseeing-real-world-fall-repository-a-large-scale-collaborative-database-to-collect-and-share-sensor-signals-from-real-world-falls/).

After getting the data, the data directory should be in the root directory. Then run [evaluation.ipynb](evaluation.ipynb), written in Python 3.11.9.