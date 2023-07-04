from stable_baselines3.common import logger

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

LOGDIR = "logs"
RESULTSPATH = 'viz/results.csv'
TMPMODELDIR = "zoo/tmp"
MODELDIR = "zoo"
DEVICE ='cuda'
logger: logger.Logger
