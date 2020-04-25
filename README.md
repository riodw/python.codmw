https://code.visualstudio.com/
https://www.python.org/ftp/python/3.7.7/python-3.7.7-amd64.exe
pip install numpy
pip install opencv-python

pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib

https://pypi.org/project/opencv-python/
https://www.microsoft.com/en-us/download/details.aspx?id=48145
Q: Import fails on Windows: ImportError: DLL load failed: The specified module could not be found.?
Windows N and KN editions do not include Media Feature Pack which is required by OpenCV. If you are using Windows N or KN edition, please install also Windows Media Feature Pack.

https://visualstudio.microsoft.com/visual-cpp-build-tools/
https://github.com/philferriere/cocoapi

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

https://github.com/protocolbuffers/protobuf/blob/master/src/README.md


```
No module named 'object_detection':

in models\research directory run the following:
python setup.py build
python setup.py install

2. go to model/research/slim and run the following:
`pip install -e .`
```