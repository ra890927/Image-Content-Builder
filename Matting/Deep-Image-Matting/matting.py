import math
import os
import random
import argparse
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def gen_trimap(alpha):
#     k_size = random.choice(range(1, 5))
#     iterations = np.random.randint(1, 20)
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
#     dilated = cv.dilate(alpha, kernel, iterations)
#     eroded = cv.erode(alpha, kernel, iterations)
#     trimap = np.zeros(alpha.shape)
#     trimap.fill(128)
#     trimap[eroded >= 255] = 255
#     trimap[dilated <= 0] = 0
#     return trimap

data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def composite4(fg, bg, a, w, h):
    print(fg.shape, bg.shape, a.shape, w, h)
    fg = np.array(fg, np.float32)
    alpha = np.expand_dims(a,axis=2)
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg


def generate_result(args):
    ckpt = torch.load(args.ckpt_point)
    model = ckpt['model'].module
    model = model.to(device)
    model.eval()
    
    img = cv.imread(args.fg)
    fg = img
    trimap = cv.imread(args.img, cv.IMREAD_GRAYSCALE)
    bg = cv.imread(args.bg)
    transformer = data_transforms['valid']
    
    # 算中心點位置 - 偏移量
    target_center = (img.shape[1] // 2 - args.p_x, img.shape[0] // 2 - args.p_y) 
    background_center = (bg.shape[1] // 2, bg.shape[0] // 2)
    
    #計算平移向量
    translation_vector = np.array(background_center) - np.array(target_center)
    #平移image
    target_translated = cv.warpAffine(img, np.float32([[1, 0, translation_vector[0]], 
                                                        [0, 1, translation_vector[1]]]), (bg.shape[1], bg.shape[0]))
    img = target_translated
    
    trimap_translated = cv.warpAffine(trimap, np.float32([[1, 0, translation_vector[0]], 
                                                        [0, 1, translation_vector[1]]]), (bg.shape[1], bg.shape[0]))
    trimap = trimap_translated
    
    h, w = img.shape[:2]
    x = torch.zeros((1, 4, h, w), dtype=torch.float)
    image = img[..., ::-1]  # RGB
    image = transforms.ToPILImage()(image)
    image = transformer(image)
    x[0:, 0:3, :, :] = image
    x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.) # trimap的size要與fg的size一致

    x = x.type(torch.FloatTensor).to(device)
    with torch.no_grad():
        pred = model(x)

    pred = pred.cpu().numpy()
    pred = pred.reshape((h, w))

    pred[trimap == 0] = 0.0
    pred[trimap == 255] = 1.0
    out = (pred.copy() * 255).astype(np.uint8)
    
    if(args.store):
        cv.imwrite(f"{args.output_folder}/pred_alpha.png", out)
    
    # 拿到alpha之後就可以做合成了          
    im, bg = composite4(img, bg, pred, w, h)
    
    if(args.store):
        cv.imwrite(f"{args.output_folder}/compose.png", im)
        
    return im
        
def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    mattingResult = generate_result(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--ckpt_point", default="./BEST_checkpoint.tar", type=str)
    parser.add_argument("--output_folder", default="./outputs")
    parser.add_argument("--fg", "--foreground", default="./0_image.png", type=str)
    parser.add_argument("--img", "--trimap", default="./0_trimap.png", type=str)
    parser.add_argument("--bg", "--background", default="./new_bg.png", type=str)
    parser.add_argument("--store", "--output_result", action='store_true')
    parser.add_argument("--p_x", "--position_x", default=200, type=int)
    parser.add_argument("--p_y", "--position_y", default=50, type=int)
    
    args = parser.parse_args()
    main(args)
    
    
    