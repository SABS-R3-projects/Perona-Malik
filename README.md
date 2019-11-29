### 
# Perona-Malik Solver

This package includes 3 three different solvers for the Perone-Malik equation
and an optimizer for retrieving the best timestep (dt) and lambda (ld) for different
images. Users are also able to change the g function (a variable function within the 
Perona-Malik equation).

The Perona-Malik equation is used to smooth images and is described in the following
link.

    http://people.maths.ox.ac.uk/trefethen/pdectb/perona2.pdf


### Prerequisites and Installing

First clone the following github.

     git clone https://github.com/SABS-R3-projects/Perona-Malik.git

Then go to the Perona-Malik directory you just cloned and 
create a virtual environment using pip or conda. The following shows how to do it
with pip. 

    python3 -m venv my_env
    
Then activate the environment and install the requirements.

    source my_env/bin/activate
    pip install -r requirements.txt
    
All prerequisities are in the requirement.txt file.

### Getting Started

The package consists of two parts. 
1. Smoothing of a picture (smoothpicture.py)
2. Optimize for the best variables to smooth picture (opt.py)

To smooth your picture you open the smoothpicture.py and manually
insert the path to your image and run the code. This will result
in three different smoothings of your picture from three different
solvers (allowing you to chose the best one). Additionally, information
about the smoothing process is also saved. You are also given the 
option to change the timesteps (dt), lambda (ld) and g function.

The three solvers consists are:
1. a finite difference (FD) solver,
2. a FD written in numpy solver,
3. and an anisotropic lib solver


To optimize for the best dt and ld for your image open the opt.py script.

All results are saved in the Results folder.

### Running the tests

Run tests by running test_image.py

### Authors
* **Brennan** 
* **Itai**
* **Simon**
* **Tobias**

### License

This project is licensed under the BSD 3-Clause License.

