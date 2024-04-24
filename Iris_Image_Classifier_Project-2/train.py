import argparse
import torch
import torch.nn as NN
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as tr 
import torchvision.datasets as dt


def train_model( data_dir , save_dir, arch='resnet18', pretrained_x=False, freeze_param =True,  learn_rate=0.01, hidden_units=512, epoches=5, optim_kind = 'SGD', use_gpu=False):
    
    # Define data transforms
    mean = [0.4363, 0.4328, 0.3291]
    std= [0.2129, 0.2075, 0.2038]


    data_transforms = {
            #2.1- transforms for the training:
            'train': tr.Compose([
                tr.Resize((224, 224)),
          tr.RandomHorizontalFlip(),
          tr.RandomRotation (10),
          tr.ToTensor(),
          tr.Normalize(torch.Tensor(mean), torch.Tensor(std))
            ]),
            #2.2- transforms for the validation, and testing sets:                      
            'test': tr.Compose([
                      tr.Resize((224, 224)),
                      tr.ToTensor(),
                      tr.Normalize(torch.Tensor(mean), torch.Tensor(std))
                ]),
            #2.3- transforms for the validation, and testing sets:                      
            'valid': tr.Compose([
                  tr.Resize((224, 224)),
                  tr.ToTensor(),
                  tr.Normalize(torch.Tensor(mean), torch.Tensor(std))
            ])
        }
    # Load the datasets and Dataloaders:

    image_datasets = {x: dt.ImageFolder(f'{data_dir}/{x}', data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'test']}

    # Select the model architecture
     #making user to select the optimizer type of:
    if arch == 'resnet18':
        model = models.resnet18(pretrained = pretrained_x)
        # Freeze parameters:
        if freeze_param:
            for param in model.parameters():
                param.requires_grad = False
        
        num_ftrs =  hidden_units
        number_of_classes = len(image_datasets['train'].classes)
        model.fc = NN.Linear(num_ftrs, number_of_classes)
        if optim_kind == 'Adam':
            optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
        elif optim_kind == 'SGD':
            optimizer = optim.SGD(model.fc.parameters(), lr=learn_rate, momentum = 0.9, weight_decay = 0.003)
    
    elif arch == 'vgg16':
        model = models.vgg16(pretrained = pretrained_x)
        # Freeze parameters
        if freeze_param:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[0].in_features
        model.classifier = NN.Sequential(
        NN.Linear(num_features, hidden_units),
        NN.ReLU(),
        NN.Dropout(0.2),
        NN.Linear(hidden_units, len(image_datasets['train'].classes))
    )
        if optim_kind == 'Adam':
            optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
        elif optim_kind == 'SGD':
            optimizer = optim.SGD(model.classifier.parameters(), lr=learn_rate, momentum = 0.9, weight_decay = 0.003)
        

    # Move the model to the appropriate device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    loss_fun = NN.CrossEntropyLoss()
    
    #storing best final testing accuracy:
    best_finl_accur = 0
   
    
    for epoch in range(epoches):
        print("Epoch number %d" %(epoch + 1))
        model.train()
        classif_loss = 0.0
        classif_correct = 0.0
        total_classif = 0
        
        for data in dataloaders['train']:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total_classif += labels.size(0)
            
            #set SGD to zero with the start of backprobagation
            optimizer.zero_grad()
            
            output = model(images)
            
            _, predicted = torch.max(output.data, 1)
            
            #setting the loss function by comparing between predicted outputs and true ones as labels
            loss = loss_fun(output, labels)
            
            #backprobagation
            loss.backward()
            optimizer.step()
            
            classif_loss += loss.item()
            classif_correct += (labels == predicted).sum().item()
            
        #tracking the performance in every epoch step:
        epoch_loss = classif_loss / len(dataloaders['train'])
        epoch_accuracy = (100 * classif_correct) /total_classif
        print("    -Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f " % (classif_correct, total_classif, epoch_accuracy, epoch_loss))
        
        #testing the model:
        epoch_accur = evaluateModel_on_testSet(model, dataloaders['test'],use_gpu)
        if epoch_accur > best_finl_accur:
            best_finl_accur = epoch_accur
            # saving model details
            #save_checkpoint(arch, model, image_datasets, n_epoch, optimizer, best_finl_accur, hidden_lyr, fl_name)
            save_checkpoint(arch, model, image_datasets, epoches, optimizer, best_finl_accur,hidden_units, save_dir)
    
    
    print("Training Finshed.")
    
    return model

# TODO: Do validation on the test set

#define the testing and evaluating function:
def evaluateModel_on_testSet(model, test_loader,use_gpu):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0.0
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max (outputs.data, 1)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
    epoch_acc = 100.0 * predicted_correctly_on_epoch / total
    print(" - Testing dataset. Got %d out of %d images correctly (%.3f%%)"%
            (predicted_correctly_on_epoch, total, epoch_acc))
    return epoch_acc

# TODO: Save the checkpoint 
def save_checkpoint(arch, model, image_datasets, n_epoch, optimizer, best_finl_accur, hidden_lyr, fl_name):
    
    state = {
        'arch': arch,
        'model_state': model.state_dict(),
        'epoches': n_epoch + 1,
        'best_accur': best_finl_accur,
        'optimizer': optimizer.state_dict(),
        'hidden_lyr': hidden_lyr,
        'class_to_idx': image_datasets['train'].class_to_idx

    }
    torch.save(state, fl_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on image data')
    parser.add_argument('data_dir', type=str,  default ='flowers' ,help='Path to the data directory')
    parser.add_argument('save_dir', type=str, default ='best_mdl_chkpoint_cpu.pth',help='Path to save the model checkpoint')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'vgg13'], help='Model architecture')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--learn_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epoches', type=int, default=5, help='Number of epochs')
    parser.add_argument('--optim', type=str, default='SGD',choices=['SGD', 'Adam'], help='kind of optimizer')
    parser.add_argument('--pre_train', type=bool, default=False , help='determine the model architechture if to be pre_trained')
    parser.add_argument('--freez', type=bool, default=True , help='determine the model parameters if to be freezed ')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()
    
    # train_model( data_dir , save_dir, arch='resnet18', pretrained_x=False, freeze_param =True
    # , learning_rate=0.001, hidden_units=512, epoches=5, optim_kind = 'SGD', use_gpu=False):
    train_model(data_dir = args.data_dir,
        save_dir = args.save_dir,
        arch = args.arch,
        pretrained_x = args.pre_train,
        freeze_param = args.freez,       
        learn_rate = args.learn_rate,
        hidden_units = args.hidden_units,
        epoches = args.epoches,
        optim_kind = args.optim,
        use_gpu = args.use_gpu
    )