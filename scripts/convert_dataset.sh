CUR_DIR=`pwd`
DATA_PATH=${CUR_DIR}/data
MODEL_PATH=${CUR_DIR}/pretrain_models/converted
# train dataset
python ${CUR_DIR}/src/reader.py  \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=512 \
    --do_lower_case="true" \
    --random_seed=1 \
    --input_file="${DATA_PATH}/train.tsv" \
    --output_file="${DATA_PATH}/train.mindrecord"

# dev dataset
python ${CUR_DIR}/src/reader.py  \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=512 \
    --do_lower_case="true" \
    --random_seed=1 \
    --input_file="${DATA_PATH}/dev.tsv" \
    --output_file="${DATA_PATH}/dev.mindrecord"

# train dataset
python ${CUR_DIR}/src/reader.py  \
    --vocab_path="${MODEL_PATH}/vocab.txt" \
    --max_seq_len=512 \
    --do_lower_case="true" \
    --random_seed=1 \
    --input_file="${DATA_PATH}/test.tsv" \
    --output_file="${DATA_PATH}/test.mindrecord"
