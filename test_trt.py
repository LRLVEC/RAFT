import sys

sys.path.append("core")

import argparse
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import glob
import numpy as np

from PIL import Image

import torch
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

from timeit import default_timer as timer

TRT_LOGGER = trt.Logger()

DEVICE = "cuda"


class TRTInfer:
    # input image is original size
    def __init__(self, engine_file_path):
        trt.init_libnvinfer_plugins(None, "")
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.initialized = False

    def init_buffers(self, shape):
        self.context = self.engine.create_execution_context()
        # shape = (shape[0], shape[1], shape[2], shape[3])
        self.context.set_binding_shape(self.engine.get_binding_index("image1"), shape)
        self.context.set_binding_shape(self.engine.get_binding_index("image2"), shape)
        self.stream = cuda.Stream()
        self.initialized = True

    def infer(self, image1, image2, output):
        self.context.execute_async_v2(
            bindings=[image1.data_ptr(), image2.data_ptr(), output.data_ptr()],
            stream_handle=self.stream.handle,
        )
        self.stream.synchronize()
        return output


def load_image(imfile, padder: InputPadder):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    if padder:
        return padder.pad(img[None])[0]
    return img[None]


def viz(img, flo, trt_flo):
    n = img.shape[0]
    for i in range(n):
        img_ = img[i].permute(1, 2, 0).cpu().numpy()
        flo_ = flo[i].permute(1, 2, 0).cpu().numpy()
        trt_flo_ = trt_flo[i].permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flo_ = flow_viz.flow_to_image(flo_)
        trt_flo_ = flow_viz.flow_to_image(trt_flo_)
        img_flo = np.concatenate([img_, flo_, trt_flo_], axis=0)

        # import matplotlib.pyplot as plt
        # plt.imshow(img_flo / 255.0)
        # plt.show()

        cv2.imshow("image", img_flo[:, :, [2, 1, 0]] / 255.0)
        cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    trtmodel = TRTInfer(args.trtmodel)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, "*.png")) + glob.glob(os.path.join(args.path, "*.jpg"))

        images = sorted(images)
        batch_size = 4
        batch_num = (len(images) - 1) // batch_size
        image0 = load_image(images[0], None)
        padder = InputPadder(image0.shape)

        for i in range(batch_num):
            image1s = torch.cat(
                [load_image(images[batch_size * i + k], padder) for k in range(batch_size)],
                0,
            ).to(DEVICE)
            image2s = torch.cat(
                [load_image(images[batch_size * i + k + 1], padder) for k in range(batch_size)],
                0,
            ).to(DEVICE)
            if trtmodel.initialized == False:
                trtmodel.init_buffers(image1s.shape)

            torch.cuda.synchronize(DEVICE)
            start = timer()
            flow_up = model(image1s, image2s, iters=20, test_mode=True)
            torch.cuda.synchronize(DEVICE)
            end = timer()

            print("Original time : %.1f ms" % (1000 * (end - start)), end="")

            output_shape = image1s.shape
            flow_up_trt = torch.empty(
                (output_shape[0], 2, output_shape[2], output_shape[3]),
                dtype=torch.float32,
                device=DEVICE,
            )

            torch.cuda.synchronize(DEVICE)
            start = timer()
            flow_up_trt = trtmodel.infer(image1s, image2s, flow_up_trt)
            torch.cuda.synchronize(DEVICE)
            end = timer()
            print(" TensorRT time : %.1f ms" % (1000 * (end - start)))

            viz(image1s, flow_up, flow_up_trt)


def model_converter(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    images = glob.glob(os.path.join(args.path, "*.png")) + glob.glob(os.path.join(args.path, "*.jpg"))
    images = sorted(images)
    image1 = load_image(images[0])
    image2 = load_image(images[1])

    print(image1.shape)
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    print(image1.shape)
    dummy_input1 = torch.randn_like(image1, device=DEVICE)
    dummy_input2 = torch.randn_like(image1, device=DEVICE)
    input_args = (
        dummy_input1,
        dummy_input2,
        {
            "iters": 20,
            "test_mode": True,
            "flow_init": None,
            "upsample": True
        },
    )
    input_names = ["image1", "image2"]
    output_names = ["flowup"]
    dynamic_axes = {
        "image1": {
            0: "batch_size",
            2: "width",
            3: "height"
        },
        "image2": {
            0: "batch_size",
            2: "width",
            3: "height"
        },
        "flowup": {
            0: "batch_size",
            2: "width",
            3: "height"
        },
    }

    torch.onnx.export(
        model,
        input_args,
        "RAFT_things_linux.onnx",
        export_params=True,
        verbose=True,
        opset_version=16,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--trtmodel", help="tensorRT model")
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args()
    # model_converter(args)
    demo(args)
