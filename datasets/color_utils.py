import cv2
from einops import rearrange
import imageio
import numpy as np
from PIL import Image, ImageDraw
import os


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, ((img + 0.055) / 1.055)**2.4, img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img > limit, 1.055 * img**(1 / 2.4) - 0.055, 12.92 * img)
    img[img > 1] = 1  # "clamp" tonemapper
    return img


def read_image(img_path, img_wh, blend_a=True, test_img_gen=False, img_id=0):
    img = imageio.imread(img_path).astype(np.float32) / 255.0

    if test_img_gen:
        print("[test after train]: img_id= {}, img = {}/{}/{}".format(
            img_id, img.shape, img.min(), img.max()))
        # save the training image
        print("saving training image to {}".format(
            'test/train_img_rep{}.jpg'.format(img_id)))
        rgb_img = Image.fromarray((255 * img).astype(np.uint8))
        rgb_img = rgb_img.convert('RGB')
        os.makedirs('./test/', exist_ok=True)
        rgb_img.save('test/train_img_rep{}.jpeg'.format(img_id))

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img


def add_perturbation(img, perturbation, seed, decent_occ=0):
    if 'color' in perturbation:
        np.random.seed(seed)
        # img_np = np.array(img) / 255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img[..., :3] = np.clip(s * img[..., :3] + b, 0, 1)
        # img = Image.fromarray((255 * img_np).astype(np.uint8))
    if 'occ' in perturbation:

        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        if decent_occ:
            left = np.random.randint(0, 600)
            top = np.random.randint(0, 600)
        else:
            left = np.random.randint(200, 400)
            top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10 * seed + i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(
                ((left + 20 * i, top), (left + 20 * (i + 1), top + 200)),
                fill=random_color)
    return img


def read_image_ngpa(img_path,
                    img_wh,
                    blend_a=True,
                    split='train',
                    t=0,
                    test_img_gen=False,
                    img_id=0):
    img = imageio.imread(img_path).astype(np.float32) / 255.0

    # add perturbations
    if t != 0 and split == 'train':  # perturb everything except the first image.
        # cf. Section D in the supplementary material
        img = add_perturbation(img, ['color'], t)

    if test_img_gen and split == 'train':
        print("[test after train]: t = {}, img_id= {}, img = {}/{}/{}".format(
            t, img_id, img.shape, img.min(), img.max()))
        # exit()
        # save the training image
        print("saving training image to {}".format(
            'test/train_img{}.jpg'.format(img_id)))
        rgb_img = Image.fromarray((255 * img).astype(np.uint8))
        rgb_img = rgb_img.convert('RGB')
        os.makedirs('./test/', exist_ok=True)
        rgb_img.save('test/train_img{}.jpeg'.format(img_id))

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img


def read_image_phototour(img_path,
                         blend_a=True,
                         test_img_gen=False,
                         img_id=0,
                         downscale=1,
                         crop_region='full'):
    img = imageio.imread(img_path).astype(np.float32) / 255.0

    if test_img_gen:
        print("[test after train]: img_id= {}, img = {}/{}/{}".format(
            img_id, img.shape, img.min(), img.max()))
        # save the training image
        print("saving training image to {}".format(
            'test/train_img_rep{}.jpg'.format(img_id)))
        rgb_img = Image.fromarray((255 * img).astype(np.uint8))
        rgb_img = rgb_img.convert('RGB')
        os.makedirs('./test/', exist_ok=True)
        rgb_img.save('test/train_img_rep{}.jpeg'.format(img_id))

    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    # height and width
    img_hw = (img.shape[0] // downscale, img.shape[1] // downscale)
    if downscale > 1:
        img = cv2.resize(img, (img_hw[1], img_hw[0]))

    if crop_region == 'left':
        img = img[:, :img_hw[1] // 2]
    elif crop_region == 'right':
        img = img[:, img_hw[1] // 2:]

    img = rearrange(img, 'h w c -> (h w) c')

    return img