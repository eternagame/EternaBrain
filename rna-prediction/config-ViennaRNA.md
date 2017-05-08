# Create Python Environment with RNA package
Download ViennaRNA (http://www.tbi.univie.ac.at/RNA/), unpack, and navigate to the folder. Then in terminal, type
```
./configure
make
sudo make install
nano .bash_profile
```
In the bash profile, type
```
export PYTHONPATH=/usr/local/lib/python2.7/site-packages:${PYTHONPATH}
Ctrl O
Return
Ctrl X
```
In the terminal:
```
source .bash_profile
```
