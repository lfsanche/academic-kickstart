---
title: nbgrader - Jupyter notebook

summary: Some links realted to
---

## General

- [short way](https://www.osc.edu/resources/getting_started/classroom_project_resource_guide/using_nbgrader_for_classroom)
- [long way - not useful](https://nbgrader.readthedocs.io/en/stable/index.html)

## My way

1. Install nbgrader

```python
conda install -c conda-forge nbgrader
```

2. Open Anaconda Powershell Prompt and change directory to the folder where you would like to store your course material

```python
nbgrader quickstart course_id

```

3. Go to the folder course_id/source/ and create folder for each problem sets.

4. Go to cmd and type
```
jupyter notebook
```
