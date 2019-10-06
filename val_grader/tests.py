from .grader import Grader, Case

import numpy as np
import torch

def get_data_loader(path_name, batch_size=1):
    
    from pathlib import Path
    from PIL import Image

    path = Path(path_name)
    
    def _loader():
        for img_path in path.glob('*.jpg'):
            img = Image.open(img_path)
            yield img

    return _loader
        
class PerceptualLoss(torch.nn.Module):
    """https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902"""
    def __init__(self, vgg):
        super().__init__()
        self.vgg_features = vgg.features
        self.layers = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        
    def forward(self, x):
        outputs = dict()
        for name, module in self.vgg_features._modules.items():
            x = module(x)
            if name in self.layers:
                outputs[self.layers[name]] = x.detach()
        return outputs


class CompressionGrader(Grader):
    """Image compression"""
    def __init__(self, *args, **kwargs):
        bottenecks = [4096,16384,65536]
        super().__init__(*args, **kwargs)

        self.encode = self.module.encode
        self.decode = self.module.decode
        
        self.scores = dict()
        for bottleneck in bottenecks:        
            l1, ssim, perceptual = self._get_performance(bottleneck)
            self.scores[bottleneck] = (l1, ssim, perceptual)
    
            print ("[%s B] L1: %.3f, SSIM: %.3f, Perceptual: %.3f"\
                %(bottleneck, np.mean(l1), np.mean(ssim), np.mean(perceptual)))
    
    
    def _get_performance(self, bottleneck):
        from itertools import combinations
        from skimage.measure import compare_ssim as _compare_ssim
        from torchvision.models import vgg
        from torchvision.transforms import functional as TF
        _vgg16 = vgg.vgg16(pretrained=True)
        _perceptual = PerceptualLoss(_vgg16).eval()
        _tensor = lambda x: TF.normalize(TF.to_tensor(x),mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])[None]
        _numpy = lambda x: np.array(x,dtype=float)/255.

        compare_l1 = lambda a, b: np.abs(_numpy(a) - _numpy(b)).mean()
        compare_ssim = lambda a, b: _compare_ssim(_numpy(a), _numpy(b), multichannel=True)
        
        l1 = []
        ssim = []
        perceptual = []

        data_loader = get_data_loader('data')
        
        def compare_perceptual(a, b):
            a_features = _perceptual(_tensor(a))
            b_features = _perceptual(_tensor(b))
            loss = 0.
            for name in a_features:
                loss += float((a_features[name] - b_features[name]).abs().mean())
            
            return loss

        for img in data_loader():
            w, h = img.size
            
            z = self.encode(img, bottleneck)
            
            assert z.nbytes <= bottleneck, "Latent vector exceeds bottleneck"
            
            img_rec = self.decode(z, bottleneck)
            
            # from PIL import Image
            # size = {4096:36,16384:73,65536:147}.get(bottleneck)
            # img_low = img.resize((size,size),Image.ANTIALIAS)
            # img_rec = img_low.resize((256,256),Image.ANTIALIAS)

            assert img_rec.size == img.size, "Decoded image has wrong resolution"
            
            rec_l1 = compare_l1(img, img_rec)
            rec_ssim = compare_ssim(img, img_rec)
            rec_perceptual = compare_perceptual(img, img_rec)

            l1.append(rec_l1)
            ssim.append(rec_ssim)
            perceptual.append(rec_perceptual)

        return l1, ssim, perceptual


    @Case(score=10)
    def test_low_l1(self, low=0.023, high=0.035):
        """4096B: L1 distance"""
        return np.clip(high-np.mean(self.scores[4096][0]), 0, high-low) / (high-low)
        
    @Case(score=10)
    def test_med_l1(self, low=0.011, high=0.023):
        """16384B: L1 distance"""
        return np.clip(high-np.mean(self.scores[16384][0]), 0, high-low) / (high-low)
    
    @Case(score=10)
    def test_high_l1(self, low=0.005, high=0.011):
        """65536B: L1 distance"""
        return np.clip(high-np.mean(self.scores[65536][0]), 0, high-low) / (high-low)
        
    @Case(score=10)
    def test_low_ssim(self, low=0.788, high=0.892):
        """4096B: SSIM"""
        return np.clip(np.mean(self.scores[4096][1])-low, 0, high-low) / (high-low)
        
    @Case(score=10)
    def test_med_ssim(self, low=0.892, high=0.971):
        """16384B: SSIM"""
        return np.clip(np.mean(self.scores[16384][1])-low, 0, high-low) / (high-low)
    
    @Case(score=10)
    def test_high_ssim(self, low=0.971, high=0.99):
        """65536B: SSIM"""
        return np.clip(np.mean(self.scores[65536][1])-low, 0, high-low) / (high-low)
        
    @Case(score=10)
    def test_low_perceptual(self, low=2.075, high=2.601):
        """4096B: perceptual"""
        return np.clip(high-np.mean(self.scores[4096][2]), 0, high-low) / (high-low)
        
    @Case(score=10)
    def test_med_perceptual(self, low=1.115, high=2.075):
        """16384B: perceptual"""
        return np.clip(high-np.mean(self.scores[16384][2]), 0, high-low) / (high-low)
    
    @Case(score=10)
    def test_high_perceptual(self, low=0.9, high=1.115):
        """65536B: perceptual"""
        return np.clip(high-np.mean(self.scores[65536][2]), 0, high-low) / (high-low)
