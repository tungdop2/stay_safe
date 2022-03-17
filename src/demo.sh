if [ -f "weights/crowdhuman_yolov5m.pt" ]; then
    echo "weight prepared."
else 
    echo "weight prepared."
    gdown --id 1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb
fi

read var var1
python3 src/demo.py video --device gpu --path $var --limit $var1