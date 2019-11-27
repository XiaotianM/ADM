import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import math
import cv2

def calcPSNR(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def unsharp_mask(image, kernel_size=(7, 7), sigma=1, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened



if __name__ == "__main__":

    GT = np.array(Image.open("gt.png"))
    LR = np.array(Image.open("resize.png"))
    # LR = unsharp_mask(LR)
    h, w = GT.shape

    GT_tensor = torch.cuda.FloatTensor(GT[np.newaxis, np.newaxis, :, :])
    LR_tensor = torch.cuda.FloatTensor(LR[np.newaxis, np.newaxis, :, :])

    GT_unfold = F.unfold(GT_tensor, (8, 8))
    LR_unfold = F.unfold(LR_tensor, (8, 8))

    GT_mean = torch.mean(GT_unfold, 1, keepdim=True)
    LR_mean = torch.mean(LR_unfold, 1, keepdim=True)

    divider = torch.sum((GT_unfold - GT_mean)**2 , 1, keepdim=True) + 1e-6
    k = torch.sum((GT_unfold - GT_mean) * (LR_unfold - LR_mean), 1, keepdim=True) / (divider)
    k = torch.clamp(k, 0, 1)
    print("K max: %f, min: %f" % (k.max(), k.min()))

    r = k * GT_unfold + (LR_mean - k * GT_mean)
    counter = F.fold(torch.ones(r.shape).cuda(), (h, w), kernel_size=(8, 8))
    Restore = F.fold(r, (h, w), kernel_size=(8, 8))

    result = Restore / counter
    result = result.squeeze().detach().cpu().numpy().astype(np.uint8)

    Image.fromarray(result).save("./adm.png")


    restore_psnr = calcPSNR(result, GT)
    LR_psnr      = calcPSNR(LR, GT)

    print("restore psnr: %f" % restore_psnr)
    print("LR psnr: %f" % LR_psnr)

