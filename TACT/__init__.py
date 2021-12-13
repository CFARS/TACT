import getpass
import logging
from logging.handlers import RotatingFileHandler
import os
from socket import gethostname
from sys import platform
from version import __version__


log_file_name = "tact.log"

try:
    username = getpass.getuser()
    hostname = gethostname()

    if platform() == 'win32':
        home_dir = r"C:/{username}/"
    else:
        home_dir = r"/home/{username}/"

    os.makedirs(os.path.join(home_dir, '.tact'))

    logfile = os.path.join(home_dir, '.tact', log_file_name)

except:
    logfile = log_file_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | [%(module)s:%(lineno)d] | %(message)s'
)

size_handler = RotatingFileHandler(logfile, maxBytes=1024*500, backupCount=4)
size_handler.setFormatter(formatter)

logger.addHandler(size_handler)
