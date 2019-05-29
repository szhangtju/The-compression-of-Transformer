echo "=== Acquiring datasets ==="
echo "---"

cd data

echo "- Downloading text8 (Character)"
mkdir -p text8
cd text8
wget --continue http://mattmahoney.net/dc/text8.zip
python ../../prep_text8.py
cd ..

echo "- Downloading enwik8 (Character)"
mkdir -p enwik8
cd enwik8
wget --continue http://mattmahoney.net/dc/enwik8.zip
wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
python prep_enwik8.py
cd ..

echo "---"
echo "Happy language modeling :)"
