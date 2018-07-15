# Builds if file is updated from the cached version
# Expects a valid ffmpeg support video
set -e # stop if error occurs


# check valid parameter number
if [ "$#" == 0 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

# break video into frames approx 15fps
ffmpeg -i $1 -r 15 output_%04d.png
mv output* img/

# for count in $(seq 1 $#); do
#     echo "\$"$count
# done # echo arguements

cd build
cmake .. 1>/dev/null
make -j4 1>/dev/null
cd ..

# run commands
for file in $(find ./img -iname 'output*.png' | sort -V); do
    echo Processing File: $file
    ./bin/hog $file $2
done

if [ -d res-img ]
then
    rm res-img
fi

mkdir res-img
mv res* res-img