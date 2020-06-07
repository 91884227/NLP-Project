# NLP-project
> [time=Mon, Jun 8, 2020 12:29 AM]

###### tags: `README` 
[TOC]

## data_preprocessing.py
### usage
```
python data_preprocessing.py SPLIT_RATE
```
| parameter | meaning | example |
| -------- | -------- |-------- | 
| SPLIT_RATE | testing set 的比率    |  0.1  |

將 `train_gold.json` 的資料拆成 `X_test.json` `X_train.json` `y_test.json` `y_train.json` 且將 label 的資料變成 `[1, 0, 0,..., 1, 0, 0]` 的形式
且輸出 label_encoding 的 class 在 `./module` 下，命名為 `label_preprocess.pkl`

### example
```
python data_preprocessing.py 0.1
```
則在`./data` 產生 `X_test.json` `X_train.json` `y_test.json` `y_train.json`
在`./module` 產生 `label_preprocess.pkl`

## bert_model.py
### usage
```
python bert_model.py PRETRAINED_MODEL_NAME BATCH_SIZE EPOCHS DEVICE ifLIMIT MAX_LENGTH ID LEARNING_RATE
```
| parameter | meaning | example |
| -------- | -------- |-------- | 
| PRETRAINED_MODEL_NAME |     | bert-base-cased |
| BATCH_SIZE |     |  2|
| EPOCHS |     |  5|
| DEVICE |     | cuda:0 |
| ifLIMIT |     |  1|
| MAX_LENGTH |     |50  |
| ID |     | test |
|  LEARNING_RATE |     | 0.0001 |

則會在 `./model_save` 儲存model。命名為 `ID_model_acc` 
acc 為 testing data 的正確率

### example
```
python bert_model.py bert-base-cased 2 5 cuda:0 1 50 test 0.0001
```
則會在 `./model_save` 儲存model。命名為 `test_model_acc` 

```
python bert_model.py bert-large-cased 2 5 cuda:0 1 50 test 0.0001
```
