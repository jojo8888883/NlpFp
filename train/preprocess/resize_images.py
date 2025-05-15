import os
import argparse
from PIL import Image


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.Resampling.LANCZOS)


def resize_images(image_roots, output_dir, size):
    """Resize the images in 'image_roots' and save into 'output_dir'."""
    for input_dir in image_roots:
        print(f"处理目录: {input_dir}")
        for idir in os.scandir(input_dir):
            if not idir.is_dir():
                continue
            # 提取原始目录名（例如 train2014, val2014）
            dirname = idir.name
            if not os.path.exists(os.path.join(output_dir, dirname)):
                os.makedirs(os.path.join(output_dir, dirname))    
            images = os.listdir(idir.path)
            n_images = len(images)
            for iimage, image in enumerate(images):
                try:
                    with open(os.path.join(idir.path, image), 'r+b') as f:
                        with Image.open(f) as img:
                            img = resize_image(img, [size, size])
                            img.save(os.path.join(output_dir, dirname, image), img.format)
                except(IOError, SyntaxError) as e:
                    print(f"处理图片出错: {os.path.join(idir.path, image)}", e)
                    pass
                if (iimage+1) % 1000 == 0:
                    print("[{}/{}] 已调整大小并保存至 '{}'."
                          .format(iimage+1, n_images, os.path.join(output_dir, dirname)))
            
            
def main(args):
    image_roots = args.image_root
    output_dir = args.save_root
    image_size = args.size
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    resize_images(image_roots, output_dir, image_size)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将VQA数据集中的图像调整到统一大小')
    
    parser.add_argument('--image_root', type=str, nargs='+', 
                        default=["/home/cyz/Data/VQA_V2/Images/train2014", 
                                "/home/cyz/Data/VQA_V2/Images/val2014"],
                        help='包含原始图像的目录列表（未调整大小的图像）')

    parser.add_argument('--save_root', type=str, default='../data_proc/images',
                        help='保存输出图像的根目录（调整大小后的图像）')

    parser.add_argument('--size', type=int, default=224,
                        help='调整大小后的图像尺寸')

    args = parser.parse_args()
    main(args)
