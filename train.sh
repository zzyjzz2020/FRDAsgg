
if [ $1 == "0" ]; then
export INSTALL_DIR=/data1/2021/zyj
cd $INSTALL_DIR
cd cocoapi/PythonAPI
python setup.py build_ext install


cd $INSTALL_DIR
cd apex
python setup.py install --cuda_ext --cpp_ext


cd $INSTALL_DIR
cd bench_exer
python setup.py build develop

unset INSTALL_DIR


elif [ $1 == "1" ]; then
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file \
"configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR \
TransformerPredictor SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD \
2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT \
checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR checkpoints/test SOLVER.BASE_LR 0.0005 \
MODEL.DEBIAS_MUTILLABEL True MODEL.FREQUENCYBRANCH True MODEL.MMLEARNING True GLOBAL_SETTING.DATASET_CHOICE VG
fi
