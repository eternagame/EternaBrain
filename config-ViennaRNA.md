# Create Python Environment with RNA package
NOTE: This process works only Python 2.

## Vienna 2.3.5
### First Installation

Download [ViennaRNA v2.3.5](http://www.tbi.univie.ac.at/RNA/), unpack, and navigate to the folder. Then in terminal, type
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

In the terminal, type:
```
source .bash_profile
```
Now, you can run Python scripts with the Vienna RNA library installed, by doing
```python
import RNA
```

### After the first Installation
Once you have executed the above commands, you do not need to execute them again,
as you have already installed the ViennaRNA package. Just type
```
source .bash_profile
```
into your terminal window, and you're good to go.

## Vienna 1.8.5
Download [Vienna v1.8.5](https://www.tbi.univie.ac.at/RNA/#download). In `predict_pm.py`, change the value of `path` to wherever the `./RNAfold` location is.
