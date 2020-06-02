import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='BioExp',  
     version='0.0.3',
     author="Avinash Kori, Parth Natekar",
     author_email="koriavinash1@gmail.com, parth@smail.iitm.ac.in",
     description="Deep Learning model analysis toolbox",
     long_description=open("README.md").read(),
     long_description_content_type="text/markdown",
     url="https://github.com/koriavinash1/BioExp",
     packages=setuptools.find_packages(),

     install_requires = [
         'pandas',
         'numpy',
         'matplotlib',
         'pillow',
         'simpleITK',
         'opencv-python',
         'tensorflow-gpu==1.14',
         'keras==2.1.5',
         'keras-vis',
         'lucid',
         'tqdm',
         'sklearn',
         'scipy>=0.19.0',
         'ppgm==0.0.3'
         ],

     classifiers=[
         "Programming Language :: Python :: 3.5",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     include_package_data=True,
     zip_safe=False,
 )
