import os
import re
path = '/Users/woody/codeup-data-science/individual-project/data/'


for filename in os.listdir(path):
    new_name = filename.replace('%27', '')
    os.rename(path + filename, path + new_name)
