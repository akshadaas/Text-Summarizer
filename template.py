import os
import logging
from pathlib import Path

logging.basicConfig(level = logging.INFO, format ='[%(asctime)s:  %(message)s:] ')

project_name = 'textSummarizer'

list_of_files  = [
    '.github/workflows/.gitkeep', #empty file is uploaded later on deleted
    f'src/{project_name}/__init__.py ', #constructor file, used when something local package needs to be imported
    f'src/{project_name}/components/__init__.py ',
    f'src/{project_name}/utils/__init__.py ',
    f'src/{project_name}/utils/common.py ',
    f'src/{project_name}/logging/__init__.py ',
    f'src/{project_name}/config/__init__.py ',
    f'src/{project_name}/config/configurations.py ',
    f'src/{project_name}/pipeline/__init__.py ',
    f'src/{project_name}/entity/__init__.py ',
    f'src/{project_name}/constants/__init__.py ',
    'config/config.yaml',
    'params.yaml',
    'app.py',
    'main.py',
    'README.md',
    'Dockerfile',
    'requirements.txt',
    'setup.py',
    'research/trails.py'
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir!= "":
        os.makedirs(filedir,exist_ok=True) #mkdir() : single dir creation, makedirs() : all intermediate dirs creation, exist_ok - error catch for intermediate dirs if present
        logging.info(f'Creating directory {filedir} for a file {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0): # if file is containing something it should not overwrite the code
        with open(filepath,'w') as f:
            pass
            logging.info(f'Creating empty file :{filename}')

    else:
        logging.info(f'File {filename} is already exists.')
