# Emotion Recognition

## Getting Started

### Prerequisites

- Python 3.10.13
- Libraries: opencv-python, scikit-learn, scikit-image, matplotlib, seaborn, numpy, pandas, tqdm, gdown


### Installation

1. Clone the repo

2. Install Dependency
```bash
pip install opencv-python scikit-learn scikit-image matplotlib seaborn numpy pandas tqdm gdown --quiet
```

### Usage

Before you training, run `download_data.py` and `data_annot.py`:
```bash
python download_data.py
python data_annot.py
```

After run `download_data.py` you will have this data structure

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


Use the `train.py` script. 
```bash
python train.py --data data/train.csv --save checkpoints --extractor haar --model mlp --test data/test.csv
python train.py --data data/train.csv --save checkpoints --extractor sift --model mlp --test data/test.csv

python train.py --data data/train.csv --save checkpoints --extractor haar --model svm --test data/test.csv
python train.py --data data/train.csv --save checkpoints --extractor sift --model svm --test data/test.csv
```

Use the `inference.py` script. 
```bash
python inference.py --image path/to/image --model path/to/file.npy
```

