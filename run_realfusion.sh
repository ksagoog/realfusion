#!/bin/bash

# --------------- SETUP ---------------------
cp $IMAGE_PATH $WORKDIR/image.png
LOCAL_IMAGE_PATH=$WORKDIR/image.png

TI_OUTPUT_DIR=$WORKDIR/ti_out
NERF_OUTPUT_DIR=$WORKDIR/nerf_out

# --------------- GET ALPHA MASK ---------------------
python scripts/extract-mask.py --image_path $LOCAL_IMAGE_PATH --output_dir $WORKDIR --overwrite --u2net_path $U2NET_PATH

# --------------- TEXTUAL INVERSION ---------------------
cd textual-inversion

python autoinit.py get_initialization $LOCAL_IMAGE_PATH --text_emb_file $TEXT_EMB_FILE

INITIALIZER_TOKEN=$(eval '(cat ${WORKDIR}/token_autoinit.txt)')

python textual_inversion.py \
  --pretrained_model_name_or_path=$SD_PATH \
  --train_data_dir=$LOCAL_IMAGE_PATH \
  --learnable_property="object" \
  --placeholder_token="_cat_statue_" \
  --initializer_token=$INITIALIZER_TOKEN \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=$TI_STEPS \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$TI_OUTPUT_DIR \
  --use_augmentations \
  --only_save_embeds
  
cd ..

# --------------- RECONSTRUCTION ---------------------
python main.py --O \
    --image_path $WORKDIR/rgba.png \
    --learned_embeds_path $WORKDIR/ti_out/learned_embeds.bin \
    --text "A high-resolution DSLR image of a <token>" \
    --workspace $NERF_OUTPUT_DIR \
    --iters $ITERS