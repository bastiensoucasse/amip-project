# AMIP Project

## Report

The report is in the `report` subfolder, containing the LaTeX sources to compile.

**Prerequisites**

- TeX Live

## Code

The actual source code is in the `src` subfolder.

## Distant Working

SSH must be configured (once) to connect to a computer from room 201 or 202. In the file `~/.ssh/config`.

```
Host cremi_proxy
    HostName jaguar.emi.u-bordeaux.fr
    User <user> # Your CREMI user name.

Host cremi_dl
    HostName <computer> # Any computer from room 201 or 202, like `ader`.
    User <user> # Your CREMI user name.
    ProxyJump cremi_proxy
```

Everytime you want to work at distance, you must turn on the computer you use from [the CREMI WOL page](https://services.emi.u-bordeaux.fr/exam-test/?page=wol)—use the Night Work Startup Mode if you work past 11p.m.

Then, simply connect from VSCode to your remote SSH Host. Don't forget to enable the environment on VSCode (interpreter at the bottom right) and on the terminal (`source /net/ens/DeepLearning-Pytorch/rtx_3060/bin/activate`).
