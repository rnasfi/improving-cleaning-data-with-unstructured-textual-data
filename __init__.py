__author__  = "MLParkerCleaner"
__status__  = "Development"
__version__ = "0.0.0"


import logging

logging.basicConfig(format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
root_logger = logging.getLogger()
gensim_logger = logging.getLogger('gensim')
root_logger.setLevel(logging.INFO)
gensim_logger.setLevel(logging.WARNING)