import gin
from experiment import run_exp

gin.parse_config_file("configs/fias.gin")
run_exp()
