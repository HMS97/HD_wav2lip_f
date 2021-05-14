
This repo is based on [wav2lip](https://github.com/Rudrabha/Wav2Lip) but finally failed.

- 1.Preprocessing
```

 python preprocess.py --data_root  ../videos/main/fps_corrected_video  --preprocessed_root ../videos/lrs2_preprocessed/

```
- 2.  Split the data
```
python split_data.py
```

- 3. train discriminator
```

python color_syncnet_train.py --data_root ../videos/oir_format_new_dataset --checkpoint_dir checkpoints --checkpoint_path checkpoints/lipsync_expert.pth

```
- 4. train the model 
``` 
python hq_wav2lip_train.py --data_root ../videos/lrs2_preprocessed/HQ_face/ --checkpoint_dir checkpoints --syncnet_checkpoint_path  checkpoints/lip_trained900.pth --checkpoint_path checkpoints/checkpoint_step000021500.pth

```
- 4. inference the model
```

python inference.py --checkpoint_path checkpoints/checkpoint_step000021500.pth --face ../videos/main/fps_corrected_video/video-3-0-4a.mp4 --audio ../dictator_orig.wav
CUDA_VISIBLE_DEVICES=1 python inference.py --checkpoint_path checkpoints/checkpoint_step000000200.pth --face ../video-0-0-1a.mov --audio ../dictator_orig.wav
```