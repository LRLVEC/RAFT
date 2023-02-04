import os
# import pycuda.driver as cuda
# cuda.init()
import tensorrt as trt


class RAFTInferTRT:

    def __init__(self, engine_file_path):
        # self.cuda_context = cuda.Device(0).make_context()
        # self.cuda_context.push()
        trt.init_libnvinfer_plugins(None, "")
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        self.logger = trt.Logger()
        with open(engine_file_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # self.stream = cuda.Stream()
        # self.cuda_context.pop()

    def init_shapes(self, shape):
        """ shape is like (batch, channel, height, width), and is already padded. """
        # elf.cuda_context.push()
        self.context.set_binding_shape(self.engine.get_binding_index("image1"), shape)
        self.context.set_binding_shape(self.engine.get_binding_index("image2"), shape)
        # self.cuda_context.pop()

    def infer_batch(self, image1, image2, output, batch_size):
        """ image1, image2, output must be continuously torch tensor on cuda device. """
        # self.cuda_context.push()
        self.init_shapes((batch_size, image1.shape[1], image1.shape[2], image1.shape[3]))
        self.context.execute_v2(
            bindings=[image1.data_ptr(), image2.data_ptr(), output.data_ptr()]
            # stream_handle=self.stream.handle,
        )
        # self.stream.synchronize()
        # self.cuda_context.pop()
        return output

    def infer(self, images, output, batch_size):
        """ images and output must be continuously torch tensor on cuda device. 
            images is a sequence.
        """
        # self.cuda_context.push()
        image_num = images.shape[0]
        if not image_num:
            return
        i = 0
        if image_num > batch_size:
            self.init_shapes((batch_size, images.shape[1], images.shape[2], images.shape[3]))
            while i < image_num - batch_size - 1:
                self.context.execute_v2(
                    bindings=[images[i].data_ptr(), images[i + 1].data_ptr(), output[i].data_ptr()]
                    # stream_handle=self.stream.handle,
                )
                i += batch_size
        left_batch = (image_num - 1) % batch_size
        if left_batch:
            self.init_shapes((left_batch, images.shape[1], images.shape[2], images.shape[3]))
            self.context.execute_v2(
                bindings=[images[i].data_ptr(), images[i + 1].data_ptr(), output[i].data_ptr()]
                # stream_handle=self.stream.handle,
            )
        # self.stream.synchronize()
        # self.cuda_context.pop()
        return output
