


```
No module named 'object_detection':

in models\research directory run the following:
python setup.py build
python setup.py install

2. go to model/research/slim and run the following:
`pip install -e .`
```

pip install tensorflow==1.15
pip install Cython contextlib2 pillow lxml jupyter matplotlib
pip install opencv-python
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI