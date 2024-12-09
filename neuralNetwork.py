import torch
from torch import nn
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "spectrograms" # change the datasets

train_dir = image_path / "train_new" # old was train
test_dir = image_path / "test_new" # should be test
test_dir_other = image_path / "words"

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(128, 128)),
    # Flip the images randomly on the horizontal
    #transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

training_data = datasets.ImageFolder(root=train_dir,# target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

testing_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

testing_data_other = datasets.ImageFolder(root=test_dir_other,
                                          transform=data_transform)

labels_map = {
    0: "ai",
    1: "ash",
    2: "d",
    3: "e",
    4: "ei",
    5: "f",
    6: "g",
    7: "k",
    8: "m",
    9: "n",
    10: "s",
    11: "schwa",
    12: "sh",
    13: "t",
    14: "v",
    15: "z",
    
}



class Sound:
    def __init__(self, sound):
        if(sound == "ai"):
            self.sound = sound
            self.vowel = True
            self.x = 0
            self.y = 3
            self.voiced = True
        elif(sound == "ash"):
            self.sound = sound
            self.vowel = True
            self.x = 0
            self.y = 2
            self.voiced = True
        elif(sound == "d"):
            self.sound = sound
            self.vowel = False
            self.x = 3
            self.y = 0
            self.voiced = True
        elif(sound == "e"):
            self.sound = sound
            self.vowel = True
            self.x = 0
            self.y = 1
            self.voiced = True
        elif(sound == "ei"):
            self.sound = sound
            self.vowel = True
            self.x = 0
            self.y = 1
            self.voiced = True
        elif(sound == "f"):
            self.sound = sound
            self.vowel = False
            self.x = 1
            self.y = 4
            self.voiced = False
        elif(sound == "g"):
            self.sound = sound
            self.vowel = False
            self.x = 7
            self.y = 0
            self.voiced = True
        elif(sound == "k"):
            self.sound = sound
            self.vowel = False
            self.x = 7
            self.y = 0
            self.voiced = False
        elif(sound == "m"):
            self.sound = sound
            self.vowel = False
            self.x = 0
            self.y = 1
            self.voiced = True
        elif(sound == "n"):
            self.sound = sound
            self.vowel = False
            self.x = 3
            self.y = 1
            self.voiced = True
        elif(sound == "s"):
            self.sound = sound
            self.vowel = False
            self.x = 3
            self.y = 4
            self.voiced = False
        elif(sound == "schwa"):
            self.sound = sound
            self.vowel = True
            self.x = 1
            self.y = 2
            self.voiced = True
        elif(sound == "sh"):
            self.sound = sound
            self.vowel = False
            self.x = 4
            self.y = 4
            self.voiced = False
        elif(sound == "t"):
            self.sound = sound
            self.vowel = False
            self.x = 3
            self.y = 0
            self.voiced = False
        elif(sound == "v"):
            self.sound = sound
            self.vowel = False
            self.x = 1
            self.y = 4
            self.voiced = True
        elif(sound == "z"):
            self.sound = sound
            self.vowel = False
            self.x = 3
            self.y = 4
            self.voiced = True
        else:
            self.sound = sound
            self.vowel = False
            self.x = 0
            self.y = 0
            self.voiced = False
        

    def distance(self, other):
        if(self.vowel == True and other.vowel == True):
            return math.sqrt((self.x- other.x) ** 2 + (self.y - other.y) ** 2)
        elif(self.vowel == False and other.vowel == False):
            #return (abs(self.x - other.x) + abs(self.y - other.y)) / 2.0
            if(self.x == other.x and self.y == other.y):
                return 0
            elif(self.x == other.x or self.y == other.y):
                return max(abs(self.x - other.x), abs(self.y - other.y))
            else:
                return 10
        else:
            return 10

    def __str__(self):
        return f"{self.sound} {self.x} {self.y}"


'''
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.permute(1,2,0).squeeze(), cmap="gray")
plt.show()
'''

batch_size = 1

# Create data loaders.
train_dataloader = DataLoader(dataset=training_data, 
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?
test_dataloader = DataLoader(dataset=testing_data, 
                             batch_size=batch_size, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data
test_dataloader_other = DataLoader(dataset=testing_data_other,
                                   batch_size=batch_size,
                                   shuffle=False)

#print(test_dataloader)

if __name__ == '__main__':
    #it = iter(test_dataloader)
    #for inputs, labels in test_dataloader:
        #print('inputs')
        #print(inputs.size())
        #print('labels')
        #print(labels)
        #print(labels.size())

    for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
            
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(3*128*128, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 16)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

        

    model = NeuralNetwork().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) # usually 1e-5
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            #print(X)
            # Compute prediction error
            pred = model(X) 
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    #print(enumerate(train_dataloader))
        
    #code below creates a model
    
    epochs = 100 # number of epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    
    torch.save(model.state_dict(), "test_layer_20_epoch.pth")
    print("Saved PyTorch Model State to test_layer_20_epoch.pth")
    

    
    #code below tests the model
    '''
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("test_layer_20_epoch.pth", weights_only=True))

    classes = training_data.classes
    #classes = testing_data_other.classes
    #print(classes[0])


    model.eval()
    #print(test_dataloader)

    
    
    size = len(test_dataloader.dataset) # make sure is right dataset
    correct = 0
    total_distance = 0
    total_distance_no_wrong = 0
    number_very_wrong = 0
    with torch.no_grad():
            for X, y in test_dataloader: # make sure is right dataset
                
                #print(X.size())
                X, y = X.to(device), y.to(device)
                pred = model(X)
                #print(pred)
                #print(y)
                predicted, actual = classes[pred[0].argmax(0)], classes[y]
                print(f'Predicted: "{predicted}", Actual: "{actual}"')
                s1 = Sound(predicted)
                s2 = Sound(actual)
                #print(s1)
                #print(s2)
                distance = s1.distance(s2)
                if(distance == 10):
                    number_very_wrong += 1
                else:
                    total_distance_no_wrong += distance
                print(distance)
                total_distance += distance
                if(predicted == actual):
                    correct += 1
    print("correct: " + str(correct))
    print("size: " + str(size))
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%")
    total_distance /= size
    #distance_no_wrong = total_distance_no_wrong / (size - number_very_wrong)
    print(f" Average Distance: {total_distance}")
    print(f" Number Very Wrong: {number_very_wrong}")
    #print(f" Average Distance Without Very Wrong: {distance_no_wrong} \n")
    '''


    '''
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("test_layer_20_epoch.pth", weights_only=True))

    classes = training_data.classes
    #classes = testing_data_other.classes
    #print(classes[0])


    model.eval()
    #print(test_dataloader)
    
    
    
    
    with torch.no_grad():
            for X, y in test_dataloader_other: # make sure is right dataset
                
                #print(X.size())
                X, y = X.to(device), y.to(device)
                pred = model(X)
                #print(pred)
                #print(y)
                predicted, actual = classes[pred[0].argmax(0)], classes[y]
                print(f'Predicted: "{predicted}", Actual: "{actual}"')
                
    '''


    
                
    

    # doesn't do anything
    '''    
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("test_model_main_100_epoch_low_lr.pth", weights_only=True))

    #classes = training_data.classes
    #classes = testing_data_other.classes
    #print(classes[0])


    model.eval()
    #print(test_dataloader)
    
    x = testing_data[0][0]
    #print(testing_data[0][0])
    #print(testing_data[0][1])
    #m = nn.Flatten()
    with torch.no_grad():
        #x.unsqueeze(0)
        #print(x.size())
        
        x = x.to(device)
        x = torch.flatten(x, start_dim=1)
        #x = x.flatten()
        #print(x.size())
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
    '''
    
