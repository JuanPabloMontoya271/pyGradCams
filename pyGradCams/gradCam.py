import numpy as np
import scipy.ndimage as nd
import torch


class GradCAM:
    
    def __init__(self, image_volume, c, features_fn, classifier_fn):
        """
        image_volume: dtype: torch.Tensor
        c: class to predict the gradcam, dtype: integer as torch.Scalar
        features_fn: feature layers from the model
        classifier_fn: classification layers from the model
        """
        return self.GradCAM(self, img, c, features_fn, classifier_fn)
    
    def GradCAM(self, img, c, features_fn, classifier_fn):
            feats = features_fn(img.cuda())
            _,N, H, W = feats.size()

            out = classifier_fn(feats)


            c_score = out[0, c]

            grads = torch.autograd.grad(c_score, feats)

            w = grads[0][0].mean(-1).mean(-1)

            sal = torch.matmul(w, feats.view(N,H*W))

            sal = sal.view(H, W).cpu().detach().numpy()
            sal = np.maximum(sal, 0)
            return sal
