import os
import sys


def block_print():
    """
    disable print statements
    """
    sys.stdout = open(os.devnull, "w")


def enable_print():
    """
    restore printing statements
    """
    sys.stdout = sys.__stdout__
