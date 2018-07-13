# Create Python Environment with RNA package

## Vienna 1.8.5
Download [Vienna v1.8.5](https://www.tbi.univie.ac.at/RNA/#download), then run

```shell
tar -zxvf ViennaRNA-1.8.5.tar.gz
cd ViennaRNA-1.8.5
./configure --prefix=/path/to/directory/ViennaRNA185
make install
```

In `predict_pm.py`, change the value of `path` to wherever you installed Vienna 1.8.5.
