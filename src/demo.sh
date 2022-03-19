if [ -f "weights/crowdhuman_yolov5m.pt" ]; then
    echo "weight prepared"
else
    mkdir -p weights
    cd weights
    echo "download weight."
    gdown --id 1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb
    cd ..
fi

python3 demo.py video --device gpu --path $1 --limit $2