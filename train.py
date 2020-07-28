import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_train_dataloader
from tqdm import tqdm

from model import model
from utils import Config

def train_model(dataloader, model, optimizer, device, num_epochs, dataset_size):
    model.to(device)

    for epoch in range(num_epochs):
        print('-' * 15)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        for phase in ['train', 'val']:   #train and validate every epoch
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for i in tqdm(range(len(dataloader[phase].dataset[0]))):
                inputs = dataloader[phase].dataset[0][i]
                #print(inputs.shape)
                labels = dataloader[phase].dataset[1][i]
                #print(labels.shape)
                inputs = inputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_size[phase]

            print('{} Loss: {:.4f} '.format(phase, epoch_loss))

    # save the model
    #saved_model = copy.deepcopy(model.state_dict())
    with open(osp.join(Config['path'],"my_model.pth"), "wb") as output_file:
        torch.save(model.state_dict(), output_file)


# load data
dataloader, dataset_size = get_train_dataloader(debug=Config['debug'], batch_size=Config['batch_size'],
                                                num_workers=Config['num_workers'])

# train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config['learning_rate'])
device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
train = train_model(dataloader, model, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)
