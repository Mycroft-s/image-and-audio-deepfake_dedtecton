from transformers import ViTFeatureExtractor

# 设置模型保存路径
model_path = "F:/FF_Dataset/outputs/final_model"

# 加载训练时用到的 feature_extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# 将 feature_extractor 保存到模型路径中，生成 preprocessor_config.json
feature_extractor.save_pretrained(model_path)

print("preprocessor_config.json has been generated in the model directory.")
