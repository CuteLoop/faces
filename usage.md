## How to find Matlab and run in HPC

### find it in HPC shared resources 
```bash
/opt/ohpc/pub/apps/matlab/r2023b
```
### add matlab to path, so it can be executed from terminal
```bash
export PATH=/opt/ohpc/pub/apps/matlab/r2023b/bin:$PATH
```
### open matlab
```bash
matlab
```
### (optional) search for mathlab instalation
```bash
find /usr/local /opt -name matlab 2>/dev/null | grep -i matlab
```
- find searches for files with name matlab in directories listed
- 2>/dev/null | grep -i matlab redirect errors and filter

## Connect HPC to GitHub via SSH

```bash
ls ~/.ssh
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```
Copy this output. this is your ssh public key. You will put this in your github profile

> settings> Access > SSH and GPG keys > New SSH key or Add SSH key.

Now try to login to github trough the HPC. clone your repository.
and verify yourself can push.

```bash
ssh -T git@github.com
git clone git@github.com:username/repository.git
cd repository
git add .
git commit -m "Your commit message"
git push
```
