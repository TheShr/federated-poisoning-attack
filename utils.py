import torch

'''
label flip logic :
    we have 3 classes..
        0 -> 1
        1 -> 2
        2 -> 0
        
'''

def train(model, loader, optimizer, loss_fn, device, poisoned=False):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if poisoned:
            y = (y +1)%3
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

def test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total
