# Assume these are set
TIME=$(eval 'date +%s')
WORKDIR=/home/jupyter/realfusion/experiments/${TIME}
mkdir -p $WORKDIR

IMAGE_PATH=/home/jupyter/cars/n04285008_10740.JPEG

U2NET_PATH="/home/jupyter/realfusion/u2net.onnx"
SD_PATH="/home/jupyter/stable-diffusion-v1-5"
TEXT_EMB_FILE="/home/jupyter/realfusion/textual-inversion/clip-vit-large-patch14-text-embeddings.pth"
CLIP_MODEL_PATH='/home/jupyter/viscam-cloud-storage-central1/kylesargent/realfusion/resources/clipmodel'
CLIP_PROCESSOR_PATH='/home/jupyter/viscam-cloud-storage-central1/kylesargent/realfusion/resources/clipprocessor'

MIDAS_REPO_PATH="/home/jupyter/.cache/torch/hub/intel-isl_MiDaS_master"
MIDAS_WEIGHTS_PATH='/home/jupyter/dpt_beit_l_512.pt'

# --------------- HPARAMS ---------------------
TI_STEPS=1
# TI_STEPS=3000
ITERS=10
# ITERS=5000
OVERRIDES='--iters 20 --HW_synthetic 256 --HW_real 256 --HW_vis 256 --lambda_mask 0.0 --lambda_entropy 0.0 --image_loss_no_opacity'