import sys
import os

class Logger(object):
    def __init__(self, path, filename):
        # Create directory if it does not exist
        os.makedirs(path, exist_ok=True)
        
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass  # Needed for Python compatibility
