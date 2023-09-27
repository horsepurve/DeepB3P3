declare -A aug
aug['raw']=''
aug['2']='_a1x2'
aug['3']='_a1x3'
aug['4']='_a1x4'
aug['5']='_a1x5'
aug['6']='_a1x6'
aug['7']='_a1x7'
aug['8']='_a1x8'
aug['9']='_a1x9'
aug['10']='_a1x10'

function do1round()
{
    # $1: augmentation key
    # $2: max length
    echo "##### aug: "${1}" | max_len: "${2}" | gpu: "$3" #####"

    dir_aug="collect/"${1} # <-- collect or collectII
    if [ ! -d $dir_aug ]; then
        echo "-> "${dir_aug}
        mkdir $dir_aug
    fi
    dir_max=${dir_aug}"/max"${2}
    if [ ! -d $dir_max ]; then
        echo "-> "${dir_max}
        mkdir $dir_max
    fi

    train_path="bbbp/d3_train"${aug[${1}]}".txt"
    test_path="bbbp/d3_test"${aug[${1}]}".txt"
    result_path=${dir_max}"/d3_test.pred.txt"
    log_path=${dir_max}"/d3_test.txt.log"

    echo ">>> train_path: "${train_path}
    echo ">>> test_path: "${test_path}
    echo ">>> result_path: "${result_path}
    echo ">>> log_path: "${log_path}
    echo ">>> max_length: "${2}
    echo ">>> gpu: "${3}

    CUDA_VISIBLE_DEVICES=${3} python DeepB3P3.py \
    --train_path $train_path \
    --test_path $test_path \
    --result_path $result_path \
    --log_path $log_path \
    --max_length ${2} \
    --conv1_kernel 10 \
    --conv2_kernel 10 \
    --regCLASS --LR 0.001 --EVALUATE_ALL --NUM_EPOCHS 50
}

aug_key='8' # 'raw' '2' '3' '4' '5' '6' '7' '8' '9' '10'
gpu="1"
do1round ${aug_key} 75 ${gpu} 
