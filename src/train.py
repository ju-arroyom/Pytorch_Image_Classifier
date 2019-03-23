import argparse
from default_network import Classifier
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def get_arguments():
    """
    Get all user defined arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="stores data directory")
    parser.add_argument("--save_dir", help="directory for saving checkpoints", type=str)
    parser.add_argument("--arch", help="specify network architecture", type=str)
    parser.add_argument("--learning_rate", help="set learning rate", type=float)
    parser.add_argument("--hidden_units", help="set units in hidden layer", type=int)
    parser.add_argument("--epochs", help="set epochs for trainining", type=int)
    parser.add_argument("--gpu", help="gpu mode on", action="store_true")
    args = parser.parse_args()
    return args

def get_input_size(model):
    """
    Get input_size from pretained network
    """
    if hasattr(model.classifier, 'hidden_layers'):
        size = model.classifier.hidden_layers[0].in_features
        return size
    else:
        first_linear = [x[1] for x in model.classifier.named_children()][0]
        size = first_linear.in_features
        return size

def default_model():
    """
    Build default model.
    """
    network = models.vgg11(pretrained=True)
    input_size = get_input_size(network)
    default_classifier =  Classifier(input_size, 102, [12544, 4096], [0.1,0.1])
    network.classifier = default_classifier
    return network

def load_data(data_dir):
    """
    Define data transforms for training
    and validation set.
    Return a dataloader dictionary.
    """
    data_transforms = {'train':transforms.Compose([transforms.RandomRotation(45),
                             transforms.RandomResizedCrop(224), 
                             transforms.RandomHorizontalFlip(), 
                             transforms.ToTensor(), 
                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                  [0.229, 0.224, 0.225])]), 
                      'valid':transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
                              transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                                        [0.229, 0.224, 0.225])])}
    image_datasets = {data: datasets.ImageFolder(data_dir+"/"+data, 
                     transform=data_transforms[data]) for data in ['train','valid']}
    
    dataloaders = {data: torch.utils.data.DataLoader(image_datasets[data], 
                   batch_size=32, shuffle = True) for data in ['train','valid']}
    return image_datasets, dataloaders

def freeze_params(model):
    """
    Freeze Network Parameters.
    """
    for param in model.parameters():
            param.requires_grad =False
    return model
    
def validation(model, validation_loader, criterion, device):
    """
    Compute loss and accuracy on the validation set.
    """
    valid_loss = 0
    accuracy = 0
    for data in validation_loader:

        images, labels = data
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def train_network(model, loaders, criterion, optimizer, epochs, device):
    """
    Train Network with parameters.
    """
    epochs = epochs
    print_cut = 40
    step = 0
    running_loss = 0
    # Apply device
    model.to(device)
    for e in range(epochs):
        model.train()
        for _, (images, labels) in enumerate(loaders['train']):
            step +=1
            images, labels = images.to(device), labels.to(device)
            # Turn-off gradient
            optimizer.zero_grad()
            # Do Forward pass
            output = model.forward(images)
            loss = criterion(output, labels)
            # Back pass
            loss.backward()
            # Optimizer step
            optimizer.step()
            # Track training loss
            running_loss += loss.item()
            # Print when you reach cut
            if step % print_cut == 0:
                # Network is on Evaluation mode
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, loaders['valid'], criterion, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f} |".format(running_loss/print_cut),
                      "Validation Loss: {:.4f} |".format(valid_loss/len(loaders['valid'])),
                      "Validation Accuracy: {:.4f}.".format(accuracy/len(loaders['valid'])))    
                running_loss = 0
                # Network is back on Training mode
                model.train()
    return model


def save_checkpoint(dir_, trained_model, epochs, architecture):
    """
    Save checkpoint dictionary.
    """
    checkpoint = {'epochs': epochs,
                  'print_cut':40,
                  'class_idx': trained_model.class_to_idx,
                  'classifier': trained_model.classifier,
                  'model_state_dict': trained_model.state_dict(),
                  'architecture': architecture
                  }
    
    torch.save(checkpoint, dir_+"/"+"model_checkpoint.pth")

if __name__ == '__main__':

    args = get_arguments()
    data_dir = args.data_dir
    images, dataloaders = load_data(data_dir)
    # Set default params
    criterion = nn.NLLLoss()
    epochs=4


    if args.arch and args.hidden_units:
        if not hasattr(eval('models.'+args.arch+"()"), 'classifier'):
            raise ValueError('Your architecture does not have a classifier attribute.')
        print(f"architecture {args.arch} and hidden_units {args.hidden_units} turned on.")
        network = eval('models.'+args.arch+"(pretrained=True)")
        network = freeze_params(network)
        input_size = get_input_size(network)
        new_classifier = Classifier(input_size, 102, [args.hidden_units])
        network.classifier = new_classifier
        architecture = args.arch
    elif args.hidden_units and not args.arch:
        print("hidden_units turned on:", args.hidden_units)
        network = default_model()
        input_size = get_input_size(network)
        new_classifier = Classifier(input_size, 102, [args.hidden_units])
        network.classifier = new_classifier
        architecture = "vgg11"
    elif args.arch:
        if not hasattr(eval('models.'+args.arch+"()"), 'classifier'):
            raise ValueError('Your architecture does not have a classifier attribute.')
        print("architecture  turned on:", args.arch)
        network = eval('models.'+args.arch+"(pretrained=True)")
        network = freeze_params(network)
        input_size = get_input_size(network)
        default_classifier = Classifier(input_size, 102, [12544, 4096], [0.1,0.1])
        network.classifier = default_classifier
        architecture = args.arch
    elif (not args.arch) and (not args.hidden_units):
        network = default_model()
        architecture = "vgg11"

    if args.learning_rate:
        print("learning_rate turned on:", args.learning_rate)
        optimizer = optim.Adam(network.classifier.parameters(), args.learning_rate)
    elif not args.learning_rate:
        optimizer = optim.Adam(network.classifier.parameters(), lr=0.001)

    if args.epochs:
        print("epochs turned on:", args.epochs)
        epochs = args.epochs

    if args.gpu:
        print("gpu turned on: True")
        trained = train_network(network, dataloaders, criterion, optimizer, epochs, 'cuda')
    else:
        trained = train_network(network, dataloaders, criterion, optimizer, epochs, 'cpu')

    trained.class_to_idx = images['train'].class_to_idx

    if args.save_dir:
        save_checkpoint(args.save_dir, trained, epochs, architecture)
   



