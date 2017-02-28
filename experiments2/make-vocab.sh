SUMPN_DATA=/proj/nlp/users/chris/sumpn-data
EXP_DATA=$SUMPN_DATA/experiments2
TRAIN_DOCS=$SUMPN_DATA/processed-content/dailymail.pcontent.train.tsv

INPUT_1E_VOCAB=$EXP_DATA/input.1e.vocab.txt
INPUT_3E_VOCAB=$EXP_DATA/input.3e.vocab.txt
OUTPUT_1E_VOCAB=$EXP_DATA/output.1e.vocab.txt
OUTPUT_3E_VOCAB=$EXP_DATA/output.3e.vocab.txt
E1_TOKENS="<E> <D> <S> <B> __UNK__ __ENT__"
E3_TOKENS="<E> <D> <S> <B> __UNK__ __PER__ __LOC__ __ORG__"

python make-input-vocab.py --documents $TRAIN_DOCS \
    --size 25000 \
    --output $INPUT_1E_VOCAB \
    --special $E1_TOKENS

python make-input-vocab.py --documents $TRAIN_DOCS \
    --size 25000 \
    --output $INPUT_3E_VOCAB \
    --special $E3_TOKENS

python make-output-vocab.py --documents $TRAIN_DOCS \
    --size 1000 \
    --output $OUTPUT_1E_VOCAB \
    --special $E1_TOKENS

python make-output-vocab.py --documents $TRAIN_DOCS \
    --size 1000 \
    --output $OUTPUT_3E_VOCAB \
    --special $E3_TOKENS
