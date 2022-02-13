# PC_motion_prediction
The point cloud baseline for moving part segmentation and motion prediction

# ENV
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install scipy
pip install h5py
pip install tensorboard
```

# Train 
```
python train.py --train_path <PATH_TO_TRAIN> --test_path <PATH_TO_TEST> --output_dir <PATH_TO_OUTPUT>
```

# Test
```
python train.py --train_path <PATH_TO_TRAIN> --test_path <PATH_TO_TEST> --output_dir <PATH_TO_OUTPUT> --test --inference_model <PATH_TO_INFERENCE_MODEL>
```

# Visualize
```
python visualize.py --result_path <PATH_TO_RESULT>
```

# Finalize_results
```
python finalize_result.py --result_path <PATH_TO_RESULT> --output <PATH_TO_OUTPUT>
```