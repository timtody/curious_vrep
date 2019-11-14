import hydra
from experiment import run_exp

@hydra.main(config_path="configs/config.yaml")
def main(cfg):
    print(cfg.pretty())
    exit(1)
    run_exp(cfg)

if __name__ == "__main__":
    main()
