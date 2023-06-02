python3 main.py --trial_name testing \
--output_folder /home/h1/yexu660b/project/results_new/testing/ \
--model_name "efficientnet_b0" \
--labeled_anno_file "/home/h1/yexu660b/project/labeled_new_modified.csv" \
--unlabeled_anno_file "/home/h1/yexu660b/project/unlabeled_new_modified.csv" \
--image_folder "/lustre/scratch2/ws/1/s7740678-data_07/raw/BM_cytomorphology_data/" \
--threshold 250 \
--ratio_label_unlabel "1:1:3" \
--stoch_depth_prob 0.3 \
--drop_out_rate 0.9
