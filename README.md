# HACAN

Official implementation of **HACAN: Hybrid Attention-Driven Cross-Layer Alignment Network for Image-Text Retrieval**.

The revised implementation uses ConvNeXt-Base and BERT to extract mid/high-level features, Local Relationship Guidance (dual-stream semantic completion), Key Semantic Filter, Global Contrastive Divergence loss, and a coarse-to-fine Hierarchical Inference Strategy (HIS).

## Setup

Python 3.8 is recommended.

```bash
pip install -r requirements.txt
```

Place the datasets under one root directory:

```text
DATA_ROOT/
├── f30k/
│   ├── images/
│   └── dataset_flickr30k.json
└── coco/
    ├── images/{train2014,val2014}/
    └── annotations/
        ├── captions_train2014.json
        ├── captions_val2014.json
        ├── coco_train_ids.npy
        ├── coco_dev_ids.npy
        └── coco_test_ids.npy
```

BERT and ImageNet-22K ConvNeXt weights are downloaded automatically when they are not cached locally.

## Training

The manuscript settings are the defaults: learning rate `1e-5`, batch size `64`, 30 epochs, margin `0.2`, backbone freezing for the first 10 epochs, and HIS `top-K=50`.

```bash
# Flickr30K
python run.py with data_root=/path/to/DATA_ROOT direction=i2t save_path=runs/f30k_i2t
python run.py with data_root=/path/to/DATA_ROOT direction=t2i save_path=runs/f30k_t2i

# MS-COCO
python run.py with coco_config data_root=/path/to/DATA_ROOT direction=i2t save_path=runs/coco_i2t
python run.py with coco_config data_root=/path/to/DATA_ROOT direction=t2i save_path=runs/coco_t2i
```

## Evaluation

```bash
python run.py with data_root=/path/to/DATA_ROOT test_only=True checkpoint=/path/to/model.ckpt direction=i2t
python run.py with coco_config data_root=/path/to/DATA_ROOT test_only=True checkpoint=/path/to/model.ckpt direction=i2t
```

For MS-COCO 1K five-fold evaluation, append `fold5=True`. The default evaluates MS-COCO 5K. Use the same `direction` as the evaluated checkpoint.

Run the lightweight implementation checks with:

```bash
python -m unittest discover -s tests -v
```
