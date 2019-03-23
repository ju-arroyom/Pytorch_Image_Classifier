import argparse
import json
import numpy as np
from PIL import Image
import torch
from torchvision import models

def load_network(checkpoint_file, device):
    """
    Load trained model.
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    arch = checkpoint['architecture']
    model_used = eval('models.'+arch+"()")
    model_used.classifier = checkpoint['classifier']
    model_used.load_state_dict(checkpoint['model_state_dict'])
    model_used.class_to_idx = checkpoint['class_idx']  
    return model_used

def load_categories(filename):
    """
    Load cat_to_name.json
    """
    with open(filename) as f:
        data = json.load(f)
    return data

def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    
    im = Image.open(image)
    d1, d2 = im.size
    # Current ratio
    org_ratio = d1 / d2

    # Keep aspect ratio
    new_size = (256 , int(256*1/org_ratio))
    im = im.resize(new_size)
    im = im.crop((0,0,224,224))

    ## Normalizing Image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(im)/255
    np_image = (np_image - mean)/std
    np_image = np_image.transpose(2,0,1)
    return np_image

def predict(image_path, model, device, topk, class_to_name=None):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """

    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze_(0).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = torch.exp(model.forward(image))[0].cpu().numpy()
    
    index_class = { v:k for k,v in model.class_to_idx.items()}
    top_k = sorted([(pr, index_class[index]) for index, pr in enumerate(output)], reverse=True)[:topk]
    pr,cl = zip(*top_k)

    if class_to_name:
        names = [class_to_name[x] for x in cl]
        return list(pr), names
    else:
       return list(pr), list(cl)

def get_arguments():
    """
    Get all user defined arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_image", help="path to image for prediction")
    parser.add_argument("checkpoint", help="checkpoint object with trained model")
    parser.add_argument("--top_k", help="return top K most likely classes", type=int)
    parser.add_argument("--category_names", help="use mapping of categories to real names")
    parser.add_argument("--gpu", help="gpu mode on", action="store_true")
    args = parser.parse_args()
    return args

def logic_pass(args, device):
    """
    Evaluates arguments for top_k and category_names
    depending on the device.
    """
    if args.top_k and args.category_names:
       pr, cl = predict(image_path, model, device, args.top_k, load_categories(args.category_names))
       print(cl, pr)
    elif args.top_k and not args.category_names:
       pr, cl = predict(image_path, model, device, args.top_k)
       print(cl, pr)
    elif not args.top_k and args.category_names:
        pr, cl = predict(image_path, model, device, 1, load_categories(args.category_names))
        print(cl, pr)
    elif not args.top_k and not args.category_names:
        pr, cl = predict(image_path, model, device, 1)
        print(cl, pr)


if __name__ == '__main__':
    
    args = get_arguments()
    image_path = args.path_to_image
    checkpoint = args.checkpoint


    if args.gpu:
        model = load_network(checkpoint, {'cuda:1':'cuda:0'})
        logic_pass(args, 'cuda')
    
    elif not args.gpu:
        model = load_network(checkpoint, 'cpu')
        logic_pass(args, 'cpu')


