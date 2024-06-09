# Emotion Recognition

## Getting Started

### Prerequisites

- Python 3.10.13
- Libraries: opencv-python, scikit-learn, scikit-image, matplotlib, seaborn, numpy, pandas, tqdm
- Download data from: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data)


### Installation

1. Clone the repo

2. Install Dependency
```bash
pip install opencv-python scikit-learn scikit-image matplotlib seaborn numpy pandas tqdm --quiet
```

### Usage

Extract data download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data) to data folder like this struct

``` bash
├── data
│   ├── test
│   │   ├── angry
│   │   ├── disgust
│   │   ├── fear
│   │   ├── happy
│   │   ├── neutral
│   │   ├── sad
│   │   └── surprise
│   └── train
│       ├── angry
│       ├── disgust
│       ├── fear
│       ├── happy
│       ├── neutral
│       ├── sad
│       └── surprise

```

Before you training, run `data_annot.py`:
```bash
python data_annot.py
```


Use the `train.py` script. 
```bash
python train.py --data path/to/train.csv --save checkpoints --test path/to/test.csv
```

Use the `inference.py` script. 
```bash
python inference.py --image path/to/image --model path/to/file.npy
```

