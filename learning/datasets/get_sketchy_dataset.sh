#!/bin/bash
fileid="0B7ISyeE8QtDdTjE1MG9Gcy1kSkE"
filename="sketchy_data.7z"
outfolder="sketchy256x256"
echo Downloading dataset...
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
echo Download complete.
echo Extracting dataset...
7z x sketchy_data.7z > /dev/null # extracts into 256x256 folder and ignores the output to prevent flood of extractions
echo Extraction complete.
mv 256x256 ${outfolder}
rm -f ./${filename}
echo Removed temporary files.
