# rssrai2019_scene_classification
Baseline: Test Acc89.8%
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

