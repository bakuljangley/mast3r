env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=6 \
  MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 \
  OMP_NUM_THREADS=8 \
  python3 visloc.py \
  --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --dataset "VislocAachenDayNight('/home/bjangley/VPR/mast3r/datasets/aachenv11/', subscene='${scene}', pairsfile='fire_top50', topk=20)" \
  --pixel_tol 5 \
  --pnp_mode poselib \
  --reprojection_error_diag_ratio 0.008 \
  --output_dir /home/bjangley/VPR/mast3r/datasets/aachenv11/output/${scene}/loc\
  