#!/bin/sh
DIR="./SentimentAnalysis"
mkdir $DIR
cd $DIR

rm -rf imdb
wget --content-disposition https://cloud.tsinghua.edu.cn/f/37bd6cb978d342db87ed/?dl=1
tar -zxvf imdb.tar.gz
rm -rf imdb.tar.gz

rm -rf SST-2
wget --content-disposition https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
unzip SST-2.zip
rm -rf SST-2.zip

cd ..
