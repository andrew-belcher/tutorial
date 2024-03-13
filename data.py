from tonic import transforms, datasets, collation

import snntorch as snn
import torch, torch.nn as nn
from torch.utils.data import DataLoader

import sinabs

# per dataset website
num_samples_train = 75466
num_samples_test = 20382
num_samples_valid = 9981
num_inputs = 700
num_hidden = 896 #variabel TWEAK
num_outputs = 35
# samples in microsecond via tonic docs
time_step = 4000 #variable TWEAK

sensor_size = datasets.SSC.sensor_size
bin_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=time_step)
dataset_train = datasets.SSC("", split='train', transform=bin_transform)
# dataset_test = datasets.SSC("", split='test', transform=bin_transform)


batch_size = 128 #variable TWEAK
train_loader = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collation.PadTensors(batch_first=True))
                        #dataset        #batchsize              #pad samples to same size / batch_first just puts batch first??? duh
# test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)



frames, targets = next(iter(train_loader))

print(frames.shape)

import torch.nn as nn


class SNN(nn.Sequential):
    def __init__(self, backend, hidden_dim: int = 896):
        super().__init__(
            nn.Linear(700, hidden_dim),
            backend.IAF(),
            nn.Linear(hidden_dim, hidden_dim),
            backend.IAF(),
            nn.Linear(hidden_dim, hidden_dim),
            backend.IAF(),
            nn.Linear(hidden_dim, 35),
        )

def training_loop(dataloader, model):
    for data, targets in iter(dataloader):
        data, targets = data.squeeze().cuda(), targets.cuda()
        sinabs.reset_states(model)
        output = model(data)
        loss = nn.functional.cross_entropy(output.sum(1), targets)
        loss.backward()


sinabs_model = SNN(sinabs.layers).cuda()

training_loop(train_loader, sinabs_model)

#128, 249, 1, 700
# batch channel height width




""" (Yes, No,
Up, Down, Left, Right, On, Off, Stop, Go, Backward, Forward,
Follow, Learn, Zero, One, Two, Three, Four, Five, Six, Seven,
Eight, Nine, |uncommon ->| Bed, Bird, Cat, Dog, Happy,
House, Marvin, Sheila, Tree, Wow) 
UNORDERED"""
