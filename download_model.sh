#!/bin/bash

FILEID=1EnqunneGzjkJqe7GmbuO7jGjM82jmNte
FILENAME=sent_model.tar.gz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o ${FILENAME}

tar -zxvf $FILENAME
rm $FILENAME
