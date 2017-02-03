
#unzip -qq neuralsum.zip 'neuralsum/dailymail/*' 

#virtualenv env
#source env/bin/activate

#pip install numpy
#pip install scipy
#pip install pandas
#pip install unidecode

#python document-reader-test.py \
#    --corpus dailymail --data-path neuralsum --output bad-paths.txt

#rm `cat bad-paths.txt | cut -f 2`

#python make-vocab.py \
#    --corpus dailymail --data-path neuralsum \
#    --max-input-sent 25 \
#    --max-highlight 4 \
#    --input-vocab data/input-vocab25k.txt --input-vocab-size 25000 \
#    --output-vocab data/output-vocab1k.txt --output-vocab-size 1000 \
#    --stats data/vocab-stats1k.txt
#
#python make-vocab.py \
#    --corpus dailymail --data-path neuralsum \
#    --max-input-sent 25 \
#    --max-highlight 4 \
#    --input-vocab data/input-vocab25k.txt --input-vocab-size 25000 \
#    --output-vocab data/output-vocab2k.txt --output-vocab-size 2000 \
#    --stats data/vocab-stats2k.txt
#
#python make-vocab.py \
#    --corpus dailymail --data-path neuralsum \
#    --max-input-sent 25 \
#    --max-highlight 4 \
#    --input-vocab data/input-vocab25k.txt --input-vocab-size 25000 \
#    --output-vocab data/output-vocab3k.txt --output-vocab-size 3000 \
#    --stats data/vocab-stats3k.txt
#
#python make-vocab.py \
#    --corpus dailymail --data-path neuralsum \
#    --max-input-sent 25 \
#    --max-highlight 4 \
#    --input-vocab data/input-vocab25k.txt --input-vocab-size 25000 \
#    --output-vocab data/output-vocab4k.txt --output-vocab-size 4000 \
#    --stats data/vocab-stats4k.txt
#
#python make-vocab.py \
#    --corpus dailymail --data-path neuralsum \
#    --max-input-sent 25 \
#    --max-highlight 4 \
#    --input-vocab data/input-vocab25k.txt --input-vocab-size 25000 \
#    --output-vocab data/output-vocab5k.txt --output-vocab-size 5000 \
#    --stats data/vocab-stats5k.txt
#
#python generate-backbone-summaries.py \
#    --corpus dailymail --data-path neuralsum --split train \
#    --max-input-sent 25 \
#    --max-highlight 4 \
#    --output data/alignments/dailymail/train 
#
#python generate-backbone-summaries.py \
#    --corpus dailymail --data-path neuralsum --split dev \
#    --max-input-sent 25 \
#    --max-highlight 4 \
#    --output data/alignments/dailymail/dev 
#
#python generate-backbone-summaries.py \
#    --corpus dailymail --data-path neuralsum --split test \
#    --max-input-sent 25 \
#    --max-highlight 4 \
#    --output data/alignments/dailymail/test 
#
#

python alignment-info.py \
    --corpus dailymail --data-path neuralsum \
    --alignments-path data/alignments \
    --max-input-sent 25 \
    --max-highlight 4 \
    --input-vocab data/input-vocab25k.txt \
    --output-vocab data/output-vocab1k.txt


