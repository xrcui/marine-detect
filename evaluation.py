from src.marine_detect.predict import predict_on_images, predict_on_video

# # Predict on a set of images using FishInv and MegaFauna models
# predict_on_images(
#     model_paths=["path/to/FishInv/model", "path/to/MegaFauna/model"],
#     confs_threshold=[0.471, 0.6],
#     images_input_folder_path="path/to/input/images",
#     images_output_folder_path="path/to/output/folder",
# )

# # Predict on a video using FishInv and MegaFauna models
# predict_on_video(
#     model_paths=["path/to/FishInv/model", "path/to/MegaFauna/model"],
#     confs_threshold=[0.471, 0.6],
#     input_video_path="path/to/input/video.mp4",
#     output_video_path="path/to/output/video.mp4",
# )

# Predict on a set of images using FishInv and MegaFauna models
predict_on_images(
    model_paths=["weights/FishInv.pt", "weights/MegaFauna.pt"],
    confs_threshold=[0.471, 0.6],
    images_input_folder_path="assets/images/input_folder",
    images_output_folder_path="assets/images/output_folder",
)

# # Predict on a video using FishInv and MegaFauna models
# predict_on_video(
#     model_paths=["path/to/FishInv/model", "path/to/MegaFauna/model"],
#     confs_threshold=[0.471, 0.6],
#     input_video_path="path/to/input/video.mp4",
#     output_video_path="path/to/output/video.mp4",
# )