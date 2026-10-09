python3 main.py --trial_name testing \
--output_folder /path/to/results/ \
--model_name "efficientnet_b0" \
--labeled_anno_file "/path/to/labeled_new_modified.csv" \
--unlabeled_anno_file "/path/to/unlabeled_new_modified.csv" \
--image_folder "/path/to/data" \
--threshold 250 \
--stoch_depth_prob 0.3 \
--drop_out_rate 0.9
