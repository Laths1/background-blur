import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from skimage import io
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet import ISNetDIS

class Mask:

    @staticmethod
    def mediapipe(im, image_path=None, model_path=r'C:\Users\lathi\Documents\projects\background blur\selfie_segmenter.tflite'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        # Create the segmenter globally
        segmenter = vision.ImageSegmenter.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im)
    
        segmentation_result = segmenter.segment(mp_image)
        
        mask = segmentation_result.category_mask.numpy_view() > 0

        return mask.astype('uint8') * 255

    @staticmethod
    def modnet(im, image_path=None, model_path=r'C:\Users\lathi\Documents\projects\background blur\modnet_photographic_portrait_matting.onnx'):

        # output_path = image_path.replace('crowd', 'crowd_masks')
        
        ref_size = 512

        # Get x_scale_factor & y_scale_factor to resize image
        def get_scale_factor(im_h, im_w, ref_size):

            if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                if im_w >= im_h:
                    im_rh = ref_size
                    im_rw = int(im_w / im_h * ref_size)
                elif im_w < im_h:
                    im_rw = ref_size
                    im_rh = int(im_h / im_w * ref_size)
            else:
                im_rh = im_h
                im_rw = im_w

            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32

            x_scale_factor = im_rw / im_w
            y_scale_factor = im_rh / im_h

            return x_scale_factor, y_scale_factor

        ##############################################
        #  Main Inference part
        ##############################################

        # read image
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

        # unify image channels to 3
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # normalize values to scale it between -1 to 1
        im = (im - 127.5) / 127.5   

        im_h, im_w, im_c = im.shape
        x, y = get_scale_factor(im_h, im_w, ref_size) 

        # resize image
        im = cv.resize(im, None, fx = x, fy = y, interpolation = cv.INTER_AREA)

        # prepare input shape
        im = np.transpose(im)
        im = np.swapaxes(im, 1, 2)
        im = np.expand_dims(im, axis = 0).astype('float32')

        # Initialize session and get prediction
        session = onnxruntime.InferenceSession(model_path, None)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: im})

        # refine matte
        matte = (np.squeeze(result[0]) * 255).astype('uint8')
        matte = cv.resize(matte, dsize=(im_w, im_h), interpolation = cv.INTER_AREA)

        
        labelled_mask = cv.threshold(matte, 127, 255, cv.THRESH_BINARY)[1]

        # cv.imwrite(output_path, labelled_mask)

        return labelled_mask

    @staticmethod
    def isnet(im, image_path=None, model_path=r'C:\Users\lathi\Documents\projects\background blur\isnet-general-use.pth'):
        input_size = [im.shape[1], im.shape[0]]
        net=ISNetDIS()
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
        net.eval()
    
        with torch.no_grad():
            
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            result=net(image)
            result=torch.squeeze(F.interpolate(result[0][0],im_shp,mode='bilinear'),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            
        return (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)

    def labelImage(img, msk, foreground=False):
        
        if not foreground:
            msk = cv.bitwise_not(msk)

        img = cv.bitwise_and(img, img, mask=msk)

        return img

    def createImageMask(img, msk, img_path):

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(msk, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        output_path = img_path.replace('crowd', 'crowd_image_and_mask')
        
        plt.tight_layout()
        plt.savefig(output_path)
    
    
   
      

        
        
        