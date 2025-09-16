from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='weather-forecasting-dnn',  
    version='1.0.0',                
    author='Akhil Haldar',     
    author_email='akhilhaldar96@gmail.com', 
    description='Weather Forecasting using Deep Neural Networks',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
