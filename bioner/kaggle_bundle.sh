#!/bin/sh
# Zip the code + data for upload as a Kaggle Dataset. Run from bioner/:  sh kaggle_bundle.sh
# zip follows the data/ symlink, so the real CSVs go inside the zip.
cd "$(dirname "$0")"
rm -f kaggle_bundle.zip
zip -rq kaggle_bundle.zip src backend requirements.txt \
    data/train/train.csv data/test/test.csv -x '*__pycache__*'
echo "wrote $(pwd)/kaggle_bundle.zip"
unzip -l kaggle_bundle.zip
