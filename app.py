import sys
import os

# Ensure the root and server directory are in the Python path
sys.path.append(os.path.abspath(os.curdir))

from server.app import app, main

if __name__ == "__main__":
    main()
