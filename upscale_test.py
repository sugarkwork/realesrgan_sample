import sys
import cv2
import os
import multiprocessing
from queue import Empty
import filelock

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer


MODE_DIR = 'models/realesrgan'


MODEL_URLS = {
    'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
}


class _ImageUpscaler(multiprocessing.Process):
    def __init__(self, queue, gpu_id, model_name) -> None:
        super().__init__()
        self.queue = queue
        self.gpu_id = gpu_id
        self.model_name = model_name
    
    def load_model(self, model_name):
        if model_name not in MODEL_URLS:
            raise ValueError(f"Model {model_name} not found in MODEL_URLS")
        
        model_path = os.path.join(MODE_DIR, model_name + '.pth')
        url = MODEL_URLS[model_name]

        if not os.path.isfile(model_path):
            lock_path = os.path.join(MODE_DIR, model_name + '.lock')
            lock = filelock.FileLock(lock_path)
            with lock:
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, MODE_DIR), progress=True, file_name=None)
        
        return model_path

    def run(self):
        # model_name: RealESRGAN_x4plus
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4

        model_path = self.load_model(self.model_name)

        # restorer
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            gpu_id=self.gpu_id,
            tile=256)
        
        while True:
            task = self.queue.get()
            if task is None:
                break
            input_path, output_path = task

            if not os.path.isfile(input_path):
                continue
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            output, _ = self.upsampler.enhance(img, outscale=4)
            cv2.imwrite(output_path, output)


class ImageUpscaler:
    def __init__(self, gpu_id=0, model_name='RealESRGAN_x4plus'):
        self.queue = multiprocessing.Queue()
        self.upscaler = _ImageUpscaler(self.queue, gpu_id, model_name)
        self.upscaler.start()

    def upscale(self, input_path, output_path):
        if input_path is None or output_path is None:
            return
        if not os.path.isfile(input_path):
            return
        self.queue.put((input_path, output_path))

    def stop(self, wait=True):
        while not wait:
            try:
                self.queue.get(block=False)
            except Empty:
                break
        self.queue.put(None)
        self.upscaler.join()


if __name__ == '__main__':
    upscaler_wrapper1 = ImageUpscaler(0)
    upscaler_wrapper2 = ImageUpscaler(1)

    upscaler_wrapper1.upscale('input1.jpg', 'output1.png')
    upscaler_wrapper2.upscale('input2.png', 'output2.png')

    upscaler_wrapper1.stop()
    upscaler_wrapper2.stop()
