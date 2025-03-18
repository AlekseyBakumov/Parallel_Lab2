# Parallel_Lab2
Compile:</br>
- cmake -Bbuild
- cd build
- make</br>


Executable files will be in build/bin/...</br>
#Usage:</br>
matrix_vector_prod N M       -> calculates product of matrix NxM and vector N on 1,2,4,7,8,16,20,40 </br>
integral                     -> calculates integral of exp(-x * x) on 1,2,4,7,8,16,20,40 </br>
iterations (single parallel) -> calculates Ax=b system on 2,4,7,8,16,20,40,60,80 </br>
iterations2 (for parallel)   -> calculates Ax=b system on 2,4,7,8,16,20,40,60,80 </br>

All programs create .cvs files with results
