import hydra
from omegaconf import DictConfig, OmegaConf
from sources.application import Application

@hydra.main(config_path="configs", config_name="simCLR")
def run(cfg: DictConfig):
    app = Application(cfg)
    print(OmegaConf.to_yaml(cfg))

    app.run()

if __name__ == "__main__":
    run()