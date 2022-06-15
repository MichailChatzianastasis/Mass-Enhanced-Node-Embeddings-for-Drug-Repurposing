[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Official Implementation for the paper <strong> Mass Enhanced Node Embeddings for Drug Repurposing </strong>.

## Abstract 
Graph representation learning has recently emerged as a promising approach to solve pharmacological tasks by modeling biological
networks. Among the different tasks, drug repurposing, the task of identifying new uses for approved or investigational drugs, has attracted a lot of attention recently. In this work, we propose a node embedding algorithm for the problem of drug repurposing. The proposed algorithm learns node representations that capture the influence of nodes in the biological network by learning a mass term for each node along with its embedding. We apply the proposed algorithm to a multiscale interactome network and embed its nodes (i. e., proteins, drugs, diseases and biological functions) into a low dimensional space. We evaluate the generated embeddings in the drug repurposing task. Our experiments show that the proposed approach outperforms the baselines and offers an improvement of 53.33% in average precision over typical walk-based embedding approaches.

<div align=center>
<img src=https://github.com/MichailChatzianastasis/Mass-Enhanced-Node-Embeddings-for-Drug-Repurposing/blob/main/figures/drug_disease.drawio.jpg width="100%">
</div>

## Data
We evaluated our proposed node embedding algorithm on a multiscale interactome dataset proposed by Camilo Ruiz et.al., 2021 (https://www.nature.com/articles/s41467-021-21770-8#Sec9).
All data is available at http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz. To download the data, please run the following code in the same directory this project is cloned to. This should result in a data/ folder populated with the relevant data.
```
wget http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz
tar -xvf data.tar.gz
```

## Contact
Please contact Michail Chatzianastasis (mixalisx97@gmail.com) for any questions.


