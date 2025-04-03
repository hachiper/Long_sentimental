python .\qurnn_for_ACLIMDB.py
| Model | cell dim | bidirectional | Acc   |
| ----- | -------- | ------------- | ----- |
| LSTM  | 300      | False         | 49.47 |
| LSTM  | 1024     | False         | 50.13 |
| LSTM  | 1024     | True          | 50.00 |

| Model | cell dim | layerNorm     | Acc   |
| ----- | -------- | ------------- | ----- |
| LSTM  | 300      | before input  | 50.00 |
| LSTM  | 300      | after input   | 50.00±0.02 |
