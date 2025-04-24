

# Spatiotemporal Generalization Graph Neural Network-based Prediction Models by Considering Morphological Diversity in Traffic Networks

> This projec aims at sovling OOD situations in zero-shot cross-region traffic prediction.
>
> 
> The original PEMS03, PEMS04, PEMS07 and PEMS08 data can be downloaded from [GitHub](https://github.com/guoshnBJTU/ASTGNN/tree/main/data).

> The whole traffic data can be partitioned into different goups, and the partitioned traffic data will be preprocessed into normalized data by Z-score normalization.


## Data preprocessing
```
python prepareData.py
```


## Prerequisites
```
pip install -r requirements.txt
```

### Training

```
python train ASTGNN.py
```

### Testing
```
python predict_ASTGNN.py
```

## The citation of paper
>@article{liu2025spatiotemporal,
  title={Spatiotemporal Generalization Graph Neural Network-Based Prediction Models by Considering Morphological Diversity in Traffic Networks},
  author={Liu, Limei and Duan, Peibo and Chen, Zhuo and Zhang, Jinghui and Feng, Siyuan and Yue, Wenwei and Wang, Yibo and Rong, Jia},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
