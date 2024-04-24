import argparse
import torch
import torch.nn as NN
import torchvision.models as models
import torchvision.transforms as tr 
from PIL import Image
import json
import numpy as np
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['arch'] == 'resnet18':
        re_model = models.resnet18(pretrained=False)
    elif checkpoint['arch'] == 'vgg13':
        re_model = models.vgg13(pretrained=False)
    else:
        raise ValueError(f'Invalid architecture: {checkpoint["arch"]}')
    
    re_model.fc = NN.Linear(checkpoint['hidden_lyr'], len(checkpoint['class_to_idx']))
    
    re_model.load_state_dict(checkpoint['model_state'])
    re_model.class_to_idx = checkpoint['class_to_idx']
    
    return re_model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        and returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # open image with PIL.Image class as PIL object:
    image = Image.open(image_path)
    
    #First, transforms  the images where the shortest side is 256 pixels, keeping the aspect ratio, center corp-244:
    image_transforms = tr.Compose([
        tr.Resize(256),
        tr.CenterCrop(224),
    ])
    processed_image = image_transforms(image)
    
    # Convert the image to a NumPy array:
    np_image = np.array(processed_image)
    
    # Normalize the image by converting from (0-255) into (0-1) values as expected for the model network:
    np_image = np_image / 255.0
    
    # Normalize the image channels
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    # Transpose the dimensions to match PyTorch's expectations as to be 
    #...first and retain the order of the other two dimensions:
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, checkpoint_path, topk=5, use_gpu=False):
    model = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load the image and preprocess it
    np_image = process_image(image_path)
    
    # Convert the NumPy array to a PyTorch tensor
    tensor_image = torch.from_numpy(np_image).float()
    
    # Add batch dimension to the tensor
    tensor_image = tensor_image.unsqueeze(0)
    
    # Move the tensor to the appropriate device (CPU or GPU)
    tensor_image = tensor_image.to(device)
    
    
    with torch.no_grad():
        output = model(tensor_image)
        probabilities = torch.softmax(output, dim = 1)
        top_probs, top_indices = torch.topk(probabilities, topk)
        
        #converting the tensors into lists:
        top_probs = top_probs.squeeze().tolist()
        top_probs = [round(prob, 8) for prob in top_probs]  # Round the probabilities to 8 decimal places
        top_indices = top_indices.squeeze().tolist()
        top_indices = list(map(str, top_indices))
        
        return top_probs, top_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict the class probabilities of an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--topk', type=int, default=5, help='Number of top most likely classes to return')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--category_names', type=str, default=True, help='Path to the JSON file containing category names')
    
    args = parser.parse_args()

    top_probs, top_indices = predict(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint_path,
        topk=args.topk,
        use_gpu=args.use_gpu,
    )
    
    category_names=args.category_names
    if category_names:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
            
    model_idx = load_checkpoint(checkpoint_path=args.checkpoint_path)
    
    #matching key labels between model_idx.class_to_idx and cat_to_name dicts:
    cat_to_name_x = {}
    for keys in cat_to_name:
        for cls in model_idx.class_to_idx:
            if keys == cls:
                cat_to_name_x[str(model_idx.class_to_idx[cls])] = cat_to_name[keys]
    
    # Get the class names corresponding to the top classes
    class_names = [cat_to_name_x[cls] for cls in top_indices]
    
        
    print(f'Top {args.topk} Probabilities:')
    for prob, class_name in zip(top_probs, class_names):
        print(f'{class_name}: {prob:.4f}')
