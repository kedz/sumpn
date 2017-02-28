if [ "$1" = "" ]; then
    echo "usage: bash train-models.sh PATH-TO-SUMPN-DATA" 
    exit;
else

    SUMPN_DATA=$1
    EXP_DATA=$SUMPN_DATA/experiments2

    INPUT_1E_VOCAB=$EXP_DATA/input.1e.vocab.txt
    OUTPUT_1E_VOCAB=$EXP_DATA/output.1e.vocab.txt

    INPUT_3E_VOCAB=$EXP_DATA/input.3e.vocab.txt
    OUTPUT_3E_VOCAB=$EXP_DATA/output.3e.vocab.txt

    TRAIN_3E_DATA=$EXP_DATA/dailymail.train.3e.data.txt
    DEV_3E_DATA=$EXP_DATA/dailymail.dev.3e.data.txt

    export LUA_PATH="$LUA_PATH;lua/?.lua"
    export OMP_NUM_THREADS=1

    th lua/generate-vis.lua \
        --data $DEV_3E_DATA \
        --model $EXP_DATA/models1_3e/model-9.bin \
        --gpu 0 \
        --seed 45345342 \
        --samples 10 \
        --vocab $OUTPUT_3E_VOCAB \
        --output $EXP_DATA/models1_3e.i9.yaml



fi
