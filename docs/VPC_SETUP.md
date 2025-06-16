# VPC Setup notes

## Script for fast setup linux gpu container ready image:

```shell
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub 
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
apt update
apt upgrade -y
apt install -y git python3-pip docker-compose
pip install poetry==2.1.3
git clone git@github.com:vodkar/llm4codesec-llm-benchmark.git
cd llm4codesec-llm-benchmark
git config user.name <your username>
git config user.email <your email>
git submodule update --init --recursive
reboot
```