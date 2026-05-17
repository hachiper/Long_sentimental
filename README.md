# PETNN for ACL-IMDB Sentiment Classification

This repository provides a simple implementation of PETNN/QURNN for ACL-IMDB sentiment classification. It is mainly used for reviewer-side testing in the response letter.

## Files

```text
qurnn_for_ACLIMDB.py    # Main running script
qurnn_ori.py            # QURNN/PETNN recurrent unit
README.md               # Instructions
````

The main script imports the QURNN model from `qurnn_ori.py`, reads the ACL-IMDB dataset from `./aclImdb/`, and loads GloVe vectors from `./glove6B/`.

## Environment

Please install the required packages:

```bash
pip install torch numpy pillow jieba gensim tensorboard thop
```

The code uses CUDA by default. If GPU is not available, please change the default CUDA setting in `qurnn_for_ACLIMDB.py`:

```python
parser.add_argument('-cuda', type=bool, default=False)
```

## Dataset

Please download the ACL-IMDB dataset and place it under the project root:

```text
aclImdb/
├── train/
│   ├── pos/
│   ├── neg/
│   └── unsup/
└── test/
    ├── pos/
    └── neg/
```

The script uses the following default paths:

```text
./aclImdb/train
./aclImdb/test
```

## GloVe Word Vectors

Please prepare the GloVe 6B 300-dimensional word vectors in word2vec format and place the file under:

```text
glove6B/
└── glove.6B.word2vec.300d.txt
```

The default path used in the script is:

```text
./glove6B/glove.6B.word2vec.300d.txt
```

## Run

After preparing the dataset and word vectors, run:

```bash
python .\qurnn_for_ACLIMDB.py
```

For Linux/macOS:

```bash
python qurnn_for_ACLIMDB.py
```

## Output

The script will train and evaluate the PETNN/QURNN-based classifier on ACL-IMDB. During training, it prints the loss and test accuracy.

Model checkpoints are saved under:

```text
./model/
```

TensorBoard logs are saved under:

```text
./log/
```

Please create these folders before running the script if they do not exist:

```bash
mkdir model
mkdir log
```

## Notes

This code is provided as a lightweight functionality check for reviewers. It is not intended to reproduce all experiments in the manuscript with a single command.

Due to differences in hardware, CUDA version, PyTorch version, and runtime environment, the exact training time and final accuracy may vary slightly.
