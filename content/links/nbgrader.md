---
title: nbgrader - Jupyter notebook

summary: Some links realted to
---

## General

- [short way](https://www.osc.edu/resources/getting_started/classroom_project_resource_guide/using_nbgrader_for_classroom)
- [long way - not useful](https://nbgrader.readthedocs.io/en/stable/index.html)

## How to  Steps

0. Install 

    ```python
    conda install -c conda-forge nbgrader
    jupyter nbextension install --sys-prefix --py nbgrader --overwrite
    jupyter nbextension enable --sys-prefix --py nbgrader
    jupyter serverextension enable --sys-prefix --py nbgrader
    ```

1. Open Anaconda Powershell Prompt and change the current directory to the one where the class files would be (e.g. home-course).

    ```
    C:\Users\your-name> cd "path-home-course"
    ```

2. Crete the course folder

    ```
    C:\Users\your-name\path-home-course> nbgrader quickstart <course101>
    ```

    Of course, you can use another course name. A folder course101 has been created into path-home-course. There are two elements: a folder source and a py file (nbgrader_config). You can drop all elements inside source folder, but NOT the folder.

    NOTE: nbgrader makes a few assumptions about how your assignment files are organized.  

            Course101/
            |- gradebook.db
            |- nbgrader_config.db
            |- source/
                |- ps1/
                    ps1.ipynb
                |- ps2/
                    ps2.ipynb                    


3. Get inside the folder course101 and launch a noteobook

    ```
    C:\Users\your-name\path-home-course\course101> jupyter notebook
    ```
    