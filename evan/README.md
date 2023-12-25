# Setup

## Environment
```bash
conda create --name IMVFX_finalProject python=3.8
conda activate IMVFX_finalProject

git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install matplotlib
```

## Usage
```bash
cd segment-anything
python SAMtest.py --img ./demo/src/assets/data/dogs.jpg --prompt_point 529 344 --prompt_label 1 --output_trimap --output_all_mask

# --img: image path
# --prompt_point: 為list, 有x,y座標(可多個)
# --prompt_label: 為list, 放是background還是foreground(可多個)
# --output_trimap: 要不要存trimap圖片(會產生在outputs)
# --output_all_mask: 要不要存SAM predict出各種分數的所有mask(會產生在outputs)
# Else: 看Customparser class
```

## Program
1. 主程式: SAMtest.py
2. Output 是 main 裡面的 cropped_image, trimap