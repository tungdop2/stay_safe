if [ -f "weights/crowdhuman_yolov5m.pt" ] || [ -f "weights/crowdhuman_yolov5s.pt" ]; then
    echo "weight prepared"
else
    echo "download weight."
    gdown --id 1SsAI7wf-hfIAO2kY9yvQWFQXADxSilMN
    unzip weights.zip
    rm weights.zip
fi

python3 demo.py video --device gpu --path $1 --limit $2