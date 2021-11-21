pip install numpy==1.17.4
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install cython
pip install pandas
pip install matplotlib
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
git clone https://github.com/pytorch/vision.git
cp vision/references/detection/*.py .
rm -rf vision
