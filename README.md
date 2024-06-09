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

Before you training, run `data_annot.py`:
```bash
python data_annot.py
```

After run `data_annot.py` you will have this data structure

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
python train.py --data path/to/train.csv --save checkpoints --test path/to/test.csv
```

Use the `inference.py` script. 
```bash
python inference.py --image path/to/image --model path/to/file.npy
```

