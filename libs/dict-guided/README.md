# Setup
Firstly, install detectron2 as well as some dependencies.

```
pip install torch==1.4.0 torchvision==0.5.0
python -m pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 weighted-levenshtein editdistance
python -m pip install dict-trie==0.0.3
python -m pip install detectron2==0.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/torch1.4/index.html

```
Next, build the source code
```
cd libs/dict-guided
python setup.py install
```
Download the pretrained model from https://drive.google.com/file/d/15rJsQCO1ewJe-EInN-V5dSCftew4vLRz/view?usp=sharing

# Inferrence
Run this command to create the detecion-recognition results. Each line in a result file has the following format: `x_1,y_1,x_2,y_2,x_3,y_3,x_4,y_4,confidence,text`

```
python demo/custom_demo.py --config-file configs/BAText/VinText/attn_R_50.yaml --input input/  --output output/ --opts MODEL.WEIGHTS trained_model.pth
```

The results can also be obtained in code using the function `create_output` in `demo/custom_demo.py`