URL="https://raw.githubusercontent.com/allenai/multimodalqa/master/dataset/MMQA_train.jsonl.gz"
OUTPUT_FILE="$./MMQA_train.jsonl.gz"

echo "Downloading MMQA_train.jsonl.gz..."
curl -L "$URL" -o "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo "Failed to download the file. Please check the URL."
    exit 1
fi

echo "Extracting MMQA_train.jsonl.gz..."
gunzip -f "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Extraction complete. File is available at $./MMQA_train.jsonl"
else
    echo "Extraction failed."
    exit 1
fi
