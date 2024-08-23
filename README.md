# Accurate and Efficient Real-World Fall Detection Using Time Series Techniques

## Abstract

Falls pose a significant health risk, particularly for older people and those with specific medical conditions. Therefore, timely fall detection is crucial for preventing fall-related complications. Existing fall detection methods often have high false alarm or false negative rates, and many rely on handcrafted features. Additionally, most approaches are evaluated using simulated falls, leading to performance degradation in real-world scenarios. This paper explores a new fall detection approach leveraging real-world fall data and state-of-the-art time series techniques. The proposed method eliminates the need for manual feature engineering and has efficient runtime. Our approach achieves high accuracy, with false alarms and false negatives each as few as one in three days on FARSEEING, a large dataset of real-world falls (mean F1 score: 90.7%). We also outperform existing methods on simulated falls datasets, FallAllD and SisFall. Furthermore, we investigate the performance of models trained on simulated data and tested on real-world data. This research presents a real-time fall detection framework with potential for real-world implementation.

## Link to full paper
You can read the full paper [here](https://www.researchgate.net/publication/382940726_Accurate_and_Efficient_Real-World_Fall_Detection_Using_Time_Series_Techniques) or download [here](paper/Accurate%20and%20Efficient%20Fall%20Detection%20Using%20Time%20Series%20Techniques.pdf). All the results can be seen in this repository, summarised in [evaluation.ipynb](evaluation.ipynb).

## Using the Code

To rerun the code, the preprocessed versions of `FallAlld` and `SisFall` used in this study can be downloaded [here](https://drive.google.com/file/d/1ysbgiGd0jDDtkTGu2HYJGiSEUawLJt8b/view?usp=sharing). The FARSEEING dataset can be requested for [here](https://farseeingresearch.eu/the-farseeing-real-world-fall-repository-a-large-scale-collaborative-database-to-collect-and-share-sensor-signals-from-real-world-falls/).

After getting the data, the data directory should be in the root directory. Then run [evaluation.ipynb](evaluation.ipynb), written in Python 3.11.9.

## Citation
`
@article{aderinola_aaltd2024,
  title={Accurate and Efficient Real-World Fall Detection Using Time Series Techniques},
  author={Aderinola, Timilehin B and Palmerini, Luca and D’Ascanio, Ilaria and Chiari, Lorenzo and Klenk, Jochen and Becker, Clemens and Caulfield, Brian and Ifrim, Georgiana}
}
`