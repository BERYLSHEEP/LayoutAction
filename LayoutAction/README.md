## Experiment command
### attension visualization
- cd LayoutAction
- Downloading the [visualization tool](https://github.com/luo3300612/Visualizer) under LayoutAction Directory: `LayoutAction/Visualizer`
- Add "from Visualizer.visualizer import get_local" in the begining of the model.py file.
```
line 18 + :from Visualizer.visualizer import get_local
```
- Add "@get_local('att_softmax')" before the forward() function of the CausalSelfAttention class in the model.py file. 
```
line 65 + : @get_local('att_softmax')
line 66: def forward(self, x, layer_past=None):
```
- script command
```
python attension_weight.py --exp attension_visualization --log_dir ./output/logs --dataset rico --split test
```
- The output resource will be saved in `output/logs/rico/attension_visualization`. The resources include the generated image, the corresponding bbox pkl file, and the attention map. 
- Naming Rules: `{it:02d}_{i:02d}_ours.png" -> "{it:02d}_{i:02d}.pt" -> "{it:02d}_{i:02d}_{token_idx}_{head_idx}.png`

### Save platte image (the color of labels in Fig.8 of the supplement)
- script command
```
python case_study.py --command save_platte --exp case_study --log_dir ./output/logs --dataset rico --split val --device 3
```
- Results are saved in `./output/logs/rico/case_study/platte_rico.png`

### Calculate labels distribution (the number of labels in Fig.8 of the supplement)
- script command
```
python case_study.py --command dataset_distribution  --exp case_study --log_dir ./output/logs --dataset rico --split val --device 3
```
- Results are saved in ./output/logs/rico/case_study/rico_label_distribution.png

### Category conditional layout generation (Fig.4 in the paper)
- script command
```
python case_study.py --command generate_ours_case  --exp case_study --log_dir ./output/logs --dataset rico --split val --device 3 --generate_type category_generate --save_num 5 --min_box_num 3
```
- Results are saved in `./output/logs/rico/case_study/category_generate`
- save_num: only save `{args.save_num}*{args.batch_size}*{multiple_runs}` images in the above directory. 
- We only save layouts which contains more than `{args.min_box_num}` elements.

### Completion layout generation (Fig.2 in the supplement)
- script command
```
python case_study.py --command generate_ours_case  --exp case_study --log_dir ./output/logs --dataset rico --split val --device 3 --generate_type completion_generate --save_num 5 --min_box_num 3
```
- Results are saved in `./output/logs/rico/case_study/completion_generate`
- save_num: only save `{args.save_num}*{args.batch_size}*{multiple_runs}` images in the above directory.
- We only save layouts which contains more than `{args.min_box_num}` elements. 

### Unconditional layout generation (Fig.1 in the supplement)
- script command
```
python case_study.py --command generate_ours_case  --exp case_study --log_dir ./output/logs --dataset rico --split val --device 3 --generate_type random_generate --save_num 5 --min_box_num 3
```
- Results are saved in ./output/logs/rico/case_study/random_generate
- save_num: only save `{args.save_num}*{args.batch_size}*{multiple_runs}` images in the above directory.
- We only save layouts which contains more than `{args.min_box_num}` elements.

### Calculate the highest iou image in the `{args.select_samples_dir}` dirctory.
- Strive the image from the training dataset with the highest IoU score.
    - The original generate type is random_generate which indicates that the origianl sample directory is `./output/logs/rico/case_study/random_generate`
    - Results are saved in `./output/logs/rico/case_study/selected`
```
python case_study.py --command cal_iou  --exp case_study --log_dir ./output/logs --dataset rico  --device 3  --select_samples_dir ./output/logs/rico/case_study/selected/ --generate_type random_generate --is_train_iou
```

- Strive the image from the generated dataset with the highest IoU score.
    - The original generate type is category_generate which indicates that the origianl sample directory is `./output/logs/rico/case_study/category_generate`
    - Results are saved in `./output/logs/rico/case_study/selected`
```
python case_study.py --command cal_iou  --exp case_study --log_dir ./output/logs --dataset rico  --device 3  --select_samples_dir ./output/logs/rico/case_study/selected/ --generate_type category_generate
```

### Evaluate the learned model for different metrics
```
python main.py --exp test --dataset publaynet --device 1 --evaluate --model_path ./output/logs/publaynet/test/checkpoints/checkpoint.pth --eval_command category_generate --save_image --calculate_coverage --calculate_probability --save_pkl
```
- `--save_image`: save images in the output/logs/publaynet/test/evaluate_samples
- `--calculate_coverage`: calculate the coverage of an image: white space/(image.width * image.height)
- `--calculate_probability`: the statistics on the action sequence (Table.1 in the supplement and Table.3 in the paper)
- `--save_pkl`: save the generated bbox for heuristic metrics(FID, IoU, Align and Overlap)
