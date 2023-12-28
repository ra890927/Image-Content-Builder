# Setup

## Environment
```bash
conda create --name IMVFX_finalProject
conda activate IMVFX_finalProject

Python 3.5.2
PyTorch 1.1.0
wget https://github.com/foamliu/Deep-Image-Matting-PyTorch/releases/download/v1.0/BEST_checkpoint.tar

pip install torchsummary
```

## Usage
```bash
cd Deep-Image-Matting
python matting.py --fg ./images/0_image.png --img ./images/0_trimap.png --bg ./images/1_new_bg.png --store

# --fg:  foreground image path
# --img: trimap path
# --bg:  background image path
# --store: save output_result
# --p_x, --position_x: object的x軸偏移量
# --p_y, --position_y: object的y軸偏移量
```

## Program
1. 主程式: matting.py
2. Output裡面是合成結果與model predict的alpha image
