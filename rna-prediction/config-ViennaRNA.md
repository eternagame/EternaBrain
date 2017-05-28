# Create Python Environment with RNA package
NOTE: Only compatible with Python 2.

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
```
Then hit ```Ctrl O```, ```Return```,  and ```Ctrl X``` to exit the bash profile editor.

In the terminal:
```
source .bash_profile
```
Now, you can run Python scripts with the Vienna RNA library installed.
