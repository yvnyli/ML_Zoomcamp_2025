from io import BytesIO
from urllib import request

from PIL import Image

import numpy as np

from onnx import load
from onnx.reference import ReferenceEvaluator


onnxfn = "hair_classifier_empty.onnx"
#onnxfn = "hair_classifier_v1.onnx"



def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


# ImageNet stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img):
    """
    img: a PIL.Image returned by prepare_image(img, target_size)
    returns: NumPy array (3, target_size) float32 normalized
    """

    # --- Convert to array + scale to [0,1] ---
    arr = np.asarray(img).astype(np.float32) / 255.0    # shape (H, W, 3)

    # --- HWC → CHW ---
    arr = np.transpose(arr, (2, 0, 1))                 # shape (3, H, W)

    # --- Normalize ---
    arr = (arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

    return arr



def predict_test():
    # load the onnx model in the image
    with open(onnxfn, "rb") as f:
        onnx_model = load(f)
    
    imurl = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
    im = download_image(imurl)
    im = prepare_image(im,(200,200))
    
    # apply imagenet normalization
    im_t = preprocess(im)
    
    # add a dimension for batch_size==1 
    im_t_b = im_t[np.newaxis, :, :, :]
    
    sess = ReferenceEvaluator(onnx_model)
    results = sess.run(None, {"input": im_t_b})
    print(results)  # display the first result
    
    return results


def lambda_handler(event, context):    
    print("Parameters:", event)
    customer = event['customer']
    prob = predict_single(customer)
    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }
    

if __name__ == "__main__":
    
    predict_test()