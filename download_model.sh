#!/bin/bash

FILEID=12Nh7U6ton4JB5K5DqOBCclpCQwsj-x5G
FILENAME=sent_model.tar.gz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o ${FILENAME}

tar -zxvf $FILENAME
rm $FILENAME

FILEID=1se1bXiSuebpphOCoZWB5MWsTShwUAsIo
FILENAME=word2idx.pkl

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o ${FILENAME}
