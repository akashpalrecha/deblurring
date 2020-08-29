cd ..
rm -rf experiments/EDSR_levin_local_test
python main.py --model_name edsr --n_colors 1 --batch_size 2 --max_epochs 1 --dataset levin --n_feats 16 --n_resblocks 4 --tag local_test

echo "-------- STARTING TEST -------"

python test.py --input_folder "sample_levin_dataset/test/blur" --experiment_folder "experiments/EDSR_levin_local_test/version_0" --output_folder "experiments/EDSR_levin_local_test/test_output"