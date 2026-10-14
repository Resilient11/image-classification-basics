import os
import torch

from torch import nn
from tqdm import tqdm

from utils import timer
from dataset import load_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_step(model, data_loader, loss_fn, optimizer: torch.optim.Optimizer):
    """
    训练

    Returns : 
        train_loss
        train_acc
    """
    model.train()
    train_acc, train_loss = 0, 0
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        
        pred = model(x)

        loss = loss_fn(pred, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logit = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        train_acc += (y == logit).sum().item() / len(pred)

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return  train_loss, train_acc


def test_step(model, data_loader, loss_fn):
    """
    测试

    Returns : 
        train_loss
        train_acc
    """
    model.eval()
    test_acc, test_loss = 0, 0
    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss
        
            test_logit = torch.argmax(pred, dim=1)
            test_acc += (y == test_logit).sum().item() / len(pred)

    test_loss = test_loss / len(data_loader)
    test_acc = test_acc / len(data_loader)
    
    return  test_loss, test_acc

    
@timer
def train(model : nn.Module, 
          train_loader : torch.utils.data.DataLoader, 
          test_loader : torch.utils.data.DataLoader, 
          loss_fn : nn.CrossEntropyLoss, 
          optimizer : torch.optim.Optimizer,
          epochs : int,
          save_dir : str):

    results = {
         'train_loss' : [],
         'test_loss' : [],
         'train_acc' : [],
         'test_acc' : []         
    }

    best_acc = 0

    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc= train_step(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_loader, loss_fn)

        print(
             f'\nEpoch: {epoch:.4f}',
             f'train_acc: {train_acc:.4f}',
             f'train_loss: {train_loss:.4f}',
             f'test_acc: {test_acc:.4f}',
             f'test_loss: {test_loss:.4f}',
        )
        
        results['train_loss'].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results['train_acc'].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results['test_acc'].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
        results['test_loss'].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)

        
        if best_acc < test_acc:
            torch.save(model.state_dict(), save_dir+'/best.pt')
    
    return results


if __name__ == "__main__":
    import tomllib

    from network import FashionMNISTModel
    from dataset import load_data
    from utils import set_seed

    with open('config.toml', 'rb') as f:
         config = tomllib.load(f)

    set_seed()

    train_loader, test_loader, class_names = load_data(config)

    model  = FashionMNISTModel(784, 10, 10)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    results = train(model,train_loader, test_loader,  loss_fn, optimizer, config['train']['epochs'], config['train']['save_dir'])
