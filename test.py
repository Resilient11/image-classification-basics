import torch
import tomllib

from network import FashionMNISTModel
from dataset import load_data


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FashionMNISTModel(784, 10, 10)
    model.to(device)

    model.load_state_dict(torch.load('./runs/best.pt'))


    with open('config.toml', "rb") as f:
        config = tomllib.load(f)

    _, test_loader, class_names = load_data(config)

    test_sample_batch = next(iter(test_loader))

    index = torch.randint(low=0, high=len(test_sample_batch[0]), size=(1,))

    test_one, test_one_label = test_sample_batch[0][index], test_sample_batch[1][index] 

    print(f'true label : {test_one_label.item()}',
        f'\ntrue class : {class_names[test_one_label.item()]}')

    test_sample_to_gpu = test_one.to(device)

    model.eval()
    with torch.inference_mode():
        output = torch.argmax(model(test_sample_to_gpu), dim=1).item()

    print(f'\npred label : {output}',
          f'\npred class : {class_names[output]}')