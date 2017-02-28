if [ "$1" = "" ]; then
    echo "usage: bash make-data.sh PATH-TO-SUMPN-DATA" 
    exit;
else

    SUMPN_DATA=$1
    EXP_DATA=$SUMPN_DATA/experiments2
    TRAIN_DOCS=$SUMPN_DATA/processed-content/dailymail.pcontent.train.tsv
    DEV_DOCS=$SUMPN_DATA/processed-content/dailymail.pcontent.dev.tsv
    TRAIN_ALIGN=$SUMPN_DATA/auto-alignment/dailymail.align.train.yaml
    DEV_ALIGN=$SUMPN_DATA/auto-alignment/dailymail.align.dev.yaml

    INPUT_1E_VOCAB=$EXP_DATA/input.1e.vocab.txt
    INPUT_3E_VOCAB=$EXP_DATA/input.3e.vocab.txt
    OUTPUT_1E_VOCAB=$EXP_DATA/output.1e.vocab.txt
    OUTPUT_3E_VOCAB=$EXP_DATA/output.3e.vocab.txt

    echo "Making training data..."
    python prepare-data.py \
        --documents $TRAIN_DOCS \
        --alignments $TRAIN_ALIGN \
        --input-vocab $INPUT_1E_VOCAB \
        --output-vocab $OUTPUT_1E_VOCAB \
        --entity-mode "1-tag" \
        --output $EXP_DATA/dailymail.train.1e.data.txt

    echo "Making training data..."
    python prepare-data.py \
        --documents $TRAIN_DOCS \
        --alignments $TRAIN_ALIGN \
        --input-vocab $INPUT_3E_VOCAB \
        --output-vocab $OUTPUT_3E_VOCAB \
        --entity-mode "3-tags" \
        --output $EXP_DATA/dailymail.train.3e.data.txt



    echo "Making development data..."
    python prepare-data.py \
        --documents $DEV_DOCS \
        --alignments $DEV_ALIGN \
        --input-vocab $INPUT_1E_VOCAB \
        --output-vocab $OUTPUT_1E_VOCAB \
        --entity-mode "1-tag" \
        --output $EXP_DATA/dailymail.dev.1e.data.txt

    echo "Making development data..."
    python prepare-data.py \
        --documents $DEV_DOCS \
        --alignments $DEV_ALIGN \
        --input-vocab $INPUT_3E_VOCAB \
        --output-vocab $OUTPUT_3E_VOCAB \
        --entity-mode "3-tags" \
        --output $EXP_DATA/dailymail.dev.3e.data.txt

fi
