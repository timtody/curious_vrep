import gin
import sys
from experiment import run_exp

logdir = sys.argv[1]
gin.parse_config_file("configs/fias.gin")
run_exp(logdir=logdir)
