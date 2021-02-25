# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_classifier.sh"
echo "for example: bash scripts/run_classifier.sh"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

mkdir -p ms_log
mkdir -p save_models
CUR_DIR=`pwd`
MODEL_PATH=${CUR_DIR}/pretrain_models
DATA_PATH=${CUR_DIR}/data
SAVE_PATH=${CUR_DIR}/save_models
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python ${CUR_DIR}/run_ernie_classifier.py  \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="true" \
    --assessment_method="Accuracy" \
    --device_id=0 \
    --epoch_num=3 \
    --num_class=3 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --train_batch_size=32 \
    --eval_batch_size=1 \
    --save_finetune_checkpoint_path="${SAVE_PATH}" \
    --load_pretrain_checkpoint_path="${MODEL_PATH}/converted/ernie.ckpt" \
    --load_finetune_checkpoint_path="" \
    --train_data_file_path="${DATA_PATH}/train.mindrecord" \
    --eval_data_file_path="${DATA_PATH}/dev.mindrecord" \
    --schema_file_path="" #> classifier_log.txt 2>&1 &
