pip install -U pip
pip install sentinelhub
pip install pandas matplotlib
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pycocotools albumentations
git clone https://github.com/pytorch/vision.git
cp vision/references/detection/*.py .
rm -rf vision