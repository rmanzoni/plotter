if test -e "$BASE_DIR"; then
    echo "BASE_DIR already set to "$BASE_DIR", exiting"
    return
fi

BASE_DIR=$PWD
echo "setting BASE_DIR to" $BASE_DIR
export BASE_DIR=$BASE_DIR # base for the code

export PYTHONPATH=$PYTHONPATH:$BASE_DIR
