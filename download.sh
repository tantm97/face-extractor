#!/bin/bash
echo "======= Download demo resource. ======="
wget -O demo.zip "https://onedrive.live.com/download?cid=FBE7F410E8BD6803&resid=FBE7F410E8BD6803%21607&authkey=ABCt59K3Xzu5PKw"

unzip demo.zip
rm demo.zip
