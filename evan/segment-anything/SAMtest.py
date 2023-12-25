from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

class CustomParser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--model_type", default="vit_h", type=str)
        self.parser.add_argument("--ckpt_point", default="./sam_vit_h_4b8939.pth", type=str)

        self.parser.add_argument(
            "--img", default="./demo/src/assets/data/dogs.jpg", type=str)
        self.parser.add_argument("--prompt_point", nargs="+", type=int, required=True)
        self.parser.add_argument("--prompt_label", nargs="+", type=int, required=True, help="label: 0 is background; 1 is foreground")
        self.parser.add_argument("--erode_iter", type=int, default=2)
        self.parser.add_argument("--dilate_iter", type=int, default=5)

        self.parser.add_argument("--output_folder", default="./outputs")
        self.parser.add_argument("--output_trimap", action='store_true')
        self.parser.add_argument("--output_all_mask", action='store_true')

    def make_parser(self):
        args = self.parser.parse_args()
        self.model_type = args.model_type
        self.ckpt_point = args.ckpt_point
        
        self.img = args.img
        self.prompt_point = args.prompt_point
        self.prompt_label = args.prompt_label
        self.erode_iter = args.erode_iter
        self.dilate_iter = args.dilate_iter

        self.output_folder = args.output_folder
        self.output_trimap = args.output_trimap
        self.output_all_mask = args.output_all_mask

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def generate_crop_boundary(image, mask):
    '''
    return: 最大可以框住整個segmentation的框的(左上x, 左上y, 右下x, 右下y)
    '''
    y_min = mask.shape[0]
    y_max = 0
    x_min = mask.shape[1]
    x_max = 0

    # mask.shape = (height, width)= (603, 1072)
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            if (mask[i][j] == False):
                continue
            y_min = min(y_min, i)
            y_max = max(y_max, i)
            x_min = min(x_min, j)
            x_max = max(x_max, j)
    
    return (y_min, x_min, y_max, x_max)

    

def generate_crop(ARGS:CustomParser):
    '''
    return: cropped_image, cropped_mask
    '''
    sam = sam_model_registry[ARGS.model_type](checkpoint=ARGS.ckpt_point)
    predictor = SamPredictor(sam)
    image = cv2.imread(ARGS.img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    prompt_point = np.array(ARGS.prompt_point).reshape((-1, 2))
    prompt_label = np.array(ARGS.prompt_label)
    masks, scores, logits = predictor.predict(prompt_point, prompt_label)

    ##  show_mask
    if (ARGS.output_all_mask):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig(f"{ARGS.output_folder}/mask{i}.jpg")
            plt.clf()

    best_mask = masks[np.argmax(scores)]
    y_min, x_min, y_max, x_max = generate_crop_boundary(image, best_mask)
    return (image[y_min:y_max+1, x_min:x_max+1], best_mask[y_min:y_max+1, x_min:x_max+1])

def generate_trimap(ARGS:CustomParser, cropped_mask):
    mask = cropped_mask.astype(np.uint8) # 變成 0 or 1 的值
    kernel = np.ones((5,5), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=ARGS.erode_iter) # Remove areas with the detections which are not obvious => make the delicate foreground part of trimap
    dilated_mask = cv2.dilate(mask, kernel, iterations=ARGS.dilate_iter) # Dilting for making the gray part of trimap    
    trimap = (mask + dilated_mask) / 2

    if (ARGS.output_trimap):
        plt.imshow(trimap, cmap="gray")
        plt.axis('off')
        plt.savefig(f"{ARGS.output_folder}/trimap.jpg")
        plt.clf()
    return trimap


if __name__ == '__main__':
    ARGS = CustomParser()
    ARGS.make_parser()
    os.makedirs(ARGS.output_folder, exist_ok=True)
    cropped_image, cropped_mask = generate_crop(ARGS)
    trimap = generate_trimap(ARGS, cropped_mask)

    # Pass "cropped_image", "trimap" to next part




    
    