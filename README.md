# rssrai2019_scene_classification

* The contest address is [here](http://rscup.bjxintong.com.cn/#/theme/1)

* Baseline:  Test Acc 90.14%

=======
- step1

```python
python3 preprocess.py
```

- step2

​		Train from scratch:

```python
python3 train.py
```

​		or Train from checkpoint:

```python
python3 train.py --net-params /path/to/*.pth
```

- step3

```python
python3 test.py
```

