python train_text_to_image.py \
  --pretrained_model_name_or_path="/mmlabworkspace/Students/visedit/ZALOAI2023/banner_advertisement/diffusion/SD_Realistic/100_epochs" \
  --train_data_dir="./train_dataset/train" \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --max_train_steps=10000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --output_dir="./finetunedmodel/SD_15" \
  --image_column="image" \
  --caption_column="caption" \
  --seed=28657