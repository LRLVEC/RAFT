import numpy as np
import cv2
import torch
import torch.nn.functional as F
from .core.utils import flow_viz


def pad_images(images):
    """ return padded images """
    ht, wd = images.shape[-2:]
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
    _pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
    return F.pad(images, _pad, mode='replicate')


def optical_flow_visualize(img, flows: list, show=True, save_video=False, save_path="video.avi"):
    """ multiple batches of flow """
    batch_size = flows[0].shape[0]
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = None
    for i in range(batch_size):
        print("frame: ", i)
        img_ = img[i].permute(1, 2, 0).cpu().numpy()
        flows_ = [flow_viz.flow_to_image(flow[i].permute(1, 2, 0).cpu().numpy()) for flow in flows]
        img_flo = np.concatenate([img_] + flows_, axis=0)
        if save_video:
            if video == None:
                shape = (img_flo.shape[1], img_flo.shape[0])
                print(shape)
                video = cv2.VideoWriter(save_path, fourcc, 30.0, shape)
            video.write(img_flo.astype(dtype=np.uint8))

        # import matplotlib.pyplot as plt
        # plt.imshow(img_flo / 255.0)
        # plt.show()
        if show:
            cv2.imshow("image", img_flo[:, :, [0, 1, 2]] / 255.0)
            cv2.waitKey(0)
    if show:
        cv2.destroyAllWindows()
    if save_video:
        video.release()

