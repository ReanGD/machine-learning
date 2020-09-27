# machine-learning

## Create virtual env

```bash
pacman -S python-tensorflow-opt-cuda python-pygame

cd ~/projects/venv
virtualenv --system-site-packages machine-learning
source machine-learning/bin/activate
pip3 install -r ~/projects/home/machine-learning/requirements.txt
deactivate
```
