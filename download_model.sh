#!/bin/bash

FILEID=1dCBNPcqPlm6Yvqa4l_fofPjSndLJ93cB
FILENAME=sent_model.tar.gz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o ${FILENAME}

tar -zxvf $FILENAME
rm $FILENAME
