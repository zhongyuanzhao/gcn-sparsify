# Distributed Link Sparsification for Scalable Scheduling using Graph Neural Networks

Paper accepted to [IEEE TWC](https://doi.org/10.1109/TWC.2025.3606741) in 2025, and conference version published in [IEEE ICASSP 2022](https://2022.ieeeicassp.org/)

Zhongyuan Zhao, Gunjan Verma, Ananthram Swami, and Santiago Segarra, " Distributed Link Sparsification for Scalable Scheduling using Graph Neural Networks," IEEE Transactions on Wireless Communications, 2025, accepted for publication, DOI: 10.1109/TWC.2025.3606741

Z. Zhao, A. Swami and S. Segarra, "Distributed Link Sparsification for Scalable Scheduling Using Graph Neural Networks," IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, Singapore, 2022, pp. 5308-5312, doi: 10.1109/ICASSP43922.2022.9747437. 


```bib
@INPROCEEDINGS{zhao2022distributed,
  author={Zhao, Zhongyuan and Swami, Ananthram and Segarra, Santiago},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Distributed Link Sparsification for Scalable Scheduling Using Graph Neural Networks}, 
  year={2022},
  month={May},
  pages={5308-5312},
  location={Singapore, Singapore},
  doi={10.1109/ICASSP43922.2022.9747437}
}
```

## Code and synthetic data

### Install Python packages

Python3.10

`pip3 install -r requirements.txt`

### Install latex for IEEE journal acceptable fonts
Also install poppler for inspecting fonts in PDF

```bash
apt-get update
apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra dvipng ghostscript cm-super
sudo apt install -y poppler-utils
```

### Datasets

The training and test data sets as listed in Table I of the journal paper are

```bash 
data/BA_Graph_Uniform_mixN_mixp_train0/ # ER training set 5970 graphs 
data/BA_Graph_Uniform_mixN_mixp_train0/ # BA training set 5970 graphs
data/ER_Graph_Uniform_GEN21_test2/ # ER test set with 500 graphs
data/BA_Graph_Uniform_GEN24_test2/ # BA test set with 860 graphs
```

### Full code and instructions to be updated soon

