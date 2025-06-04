## Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals

Deep neural networks have been successfully applied to problems in many domains. Understanding their inner workings with respect to feature selection and decision making, however, remains challenging and thus trained models are often regarded as black boxes. Layerwise Relevance Propagation (LRP) addresses this issue by finding those features that a model relies on, offering deeper understanding and interpretation of trained networks. This repository contains code and data used in **Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals** ([https://www.sciencedirect.com/science/article/pii/S0016003223007536](https://www.sciencedirect.com/science/article/pii/S0016003223007536)).

### Repository structure

#### data (audioMNIST)
* The dataset consists of 30000 audio samples of spoken digits (0-9) of 60 different speakers. 
* There is one directory per speaker holding the audio recordings. 
* Additionally "audioMNIST_meta.txt" provides meta information such as gender or age of each speaker.

#### models
* There are two different model architectures and training parameters in the CAFFE deep learning framework format.
* Bash script to train and test models.

#### recording_scripts
* Scripts to gather further audio samples. 

#### preprocessing_data.py
* A python script to preprocess the provided audio records and to store them in a format suitable for the provided caffe models.


### Reference
If you use the provided audioMNIST dataset for your project, please cite [our paper](https://www.sciencedirect.com/science/article/pii/S0016003223007536):

```bib
@article{audiomnist2023,
    title = {AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark},
    journal = {Journal of the Franklin Institute},
    year = {2023},
    issn = {0016-0032},
    doi = {https://doi.org/10.1016/j.jfranklin.2023.11.038},
    url = {https://www.sciencedirect.com/science/article/pii/S0016003223007536},
    author = {Sören Becker and Johanna Vielhaben and Marcel Ackermann and Klaus-Robert Müller and Sebastian Lapuschkin and Wojciech Samek},
    keywords = {Deep learning, Neural networks, Interpretability, Explainable artificial intelligence, Audio classification, Speech recognition},
}
```

(The above paper is the published (and extended) version of the paper that was previously on arxiv under the title _Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals_.)
