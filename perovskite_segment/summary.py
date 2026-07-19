import torch
from torchsummary import summary
from models import get_model
import hydra

@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
def main(cfg):
    printsummary(cfg)


def printsummary(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = get_model(cfg, device)
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    summary(mdl, input_size=(3, 256, 256), batch_size=cfg.dataset.batch_size)


if __name__ == '__main__':
    main()