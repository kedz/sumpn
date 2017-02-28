if [ "$1" = "" ]; then
    echo "usage: bash train-models.sh PATH-TO-SUMPN-DATA" 
    exit;
else

    SUMPN_DATA=$1
    EXP_DATA=$SUMPN_DATA/experiments2

    TRAIN_3E_DATA=$EXP_DATA/dailymail.train.3e.data.txt
    DEV_3E_DATA=$EXP_DATA/dailymail.dev.3e.data.txt
    INPUT_1E_VOCAB=$EXP_DATA/input.1e.vocab.txt
    INPUT_3E_VOCAB=$EXP_DATA/input.3e.vocab.txt
    OUTPUT_1E_VOCAB=$EXP_DATA/output.1e.vocab.txt
    OUTPUT_3E_VOCAB=$EXP_DATA/output.3e.vocab.txt

    export LUA_PATH="$LUA_PATH;lua/?.lua"
    export OMP_NUM_THREADS=1

    th lua/train-model1.lua --input-vocab $INPUT_3E_VOCAB \
        --output-vocab $OUTPUT_3E_VOCAB \
        --data $TRAIN_3E_DATA \
        --save $EXP_DATA/models1_3e \
        --batch-size 15 \
        --dims 300 \
        --lr .0001 \
        --seed 1986 \
        --gpu 1 \
        --epochs 50

fi
