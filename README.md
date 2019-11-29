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

To optimize for the best dt and ld for your image open the opt.py script.


### Running the tests

Run tests by running test_image.py

### Authors
* **Brennan** 
* **Itai**
* **Simon**
* **Tobias**

### License

This project is licensed under the BSD 3-Clause License.

