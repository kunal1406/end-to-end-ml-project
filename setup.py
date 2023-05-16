from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements

    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n","") for req in requirements ]

        if "-e ." in requirements:
            requirements.remove("-e .")


setup(

name='end-to-end-ml-project',
version = '0.0.1',
author = 'Kunal',
author_email = 'mehtakunalg@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')

)