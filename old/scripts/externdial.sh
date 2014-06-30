#! /bin/bash
#
# Logs into the server

echo "Please note that you have to connect to the TU-network via vpn! Press Ctrl+C to abort at any time."
echo "Enter RZ Username:"
read USER

echo "Enter mount point:"
read MP
if [ ! -d "$MP" ]; then
  echo "Directory doesn't exist and will be created..."
  mkdir ${MP}
fi

echo "Do you want to use NVIDIA (n) or ATI (a) GPU? (n/a)"
read answer
if [ "$answer" == "n" ]; then
  GPU=xy5.ti6.tu-harburg.de
elif [ "$answer" == "a" ]; then
  GPU=xy3.ti6.tu-harburg.de
else
  echo "Please type n or a!"
  exit
fi

PORT=2126
GPU_USER=stud1

echo "Attempting ssh port forwarding stuff..."
echo "ssh -N -f -L ${PORT}:${GPU}:22 ${USER}@ssh.rz.tu-harburg.de"
ssh -N -f -L ${PORT}:${GPU}:22 ${USER}@ssh.rz.tu-harburg.de

echo "Attempting to mount fs..."
echo "sshfs ${GPU_USER}@localhost:/mounts/student/${GPU_USER} ${MP} -p ${PORT} -o workaround=all -o TCPKeepAlive=yes"
sshfs ${GPU_USER}@localhost:/mounts/student/${GPU_USER} ${MP} -p ${PORT} -o workaround=all -o TCPKeepAlive=yes

echo "Attempting to open ssh console..."
echo "ssh -p ${PORT} ${GPU_USER}@localhost -XC"
ssh -p ${PORT} ${GPU_USER}@localhost -XC

