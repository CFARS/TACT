import getpass
import logging
from logging.handlers import RotatingFileHandler
import os
from socket import gethostname
from sys import platform


try:
    log_file_name = "tactlog.log"
    username = getpass.getuser()
    hostname = gethostname()

    if platform() == 'win32':
        home_dir = "C:/{username}/"
    else:
        home_dir = "/home/{username}/"

    os.makedirs(os.path.join(home_dir, '.tact'))

    logfile = os.path.join(home_dir, '.tact', log_file_name)

except:
    logfile = log_file_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | [%(module)s:%(lineno)d] | %(message)s'
)

size_handler = RotatingFileHandler(log_file, maxBytes=1024*1000, backupCount=4)
size_handler.setFormatter(formatter)

logger.addHandler(size_handler)
