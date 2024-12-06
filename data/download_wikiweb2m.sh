mkdir -p data/wikiweb2m

base_url="https://storage.googleapis.com/gresearch/wit/wikiweb2m"
files=(
    "wikiweb2m-train.tfrecord.gz-00000-of-00005"
    "wikiweb2m-train.tfrecord.gz-00001-of-00005"
    "wikiweb2m-train.tfrecord.gz-00002-of-00005"
    "wikiweb2m-train.tfrecord.gz-00003-of-00005"
    "wikiweb2m-train.tfrecord.gz-00004-of-00005"
)

for file in "${files[@]}"; do
    wget -P data/wikiweb2m "${base_url}/${file}"
done

echo "Download complete. Files saved in data/wikiweb2m"