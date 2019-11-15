import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='BioExp',  
     version='0.0.1',
     author="Avinash Kori, Parth Natekar",
     author_email="koriavinash1@gmail.com, parth@smail.iitm.ac.in",
     description="Deep Learning model analysis toolbox",
     long_description=open("README.md").read(),
     long_description_content_type="text/markdown",
     url="https://github.com/parthnatekar/BioExp",
     packages=setuptools.find_packages(),

     install_requires = [
         'pandas',
         'numpy',
         'scipy==1.6.0',
         'matplotlib',
         'pillow',
         'simpleITK',
         'opencv-python',
         'tensorflow-gpu==1.14',
         'keras',
         'keras-vis',
         'lucid'
         ],

     classifiers=[
         "Programming Language :: Python :: 3.5",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     include_package_data=True,
     zip_safe=False,
 )
