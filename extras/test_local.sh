cd ..

echo "---- Deleting previous experiment ----"
echo "rm -rf experiments/EDSR_levin_local_test"
rm -rf experiments/EDSR_levin_local_test
echo "---- DONE ----"
echo ""

python main.py --model_name edsr --n_colors 1 --batch_size 2 --max_epochs 2 --dataset levin --n_feats 16 --n_resblocks 4 --tag local_test --lr_decay_every_n_epochs 1 --lr_decay_factor 0.5

echo ""
echo ""
echo "-------- STARTING TEST -------"
echo ""
echo ""

python test.py --input_folder "sample_levin_dataset/test/blur" --experiment_folder "experiments/EDSR_levin_local_test/version_0" --output_folder "experiments/EDSR_levin_local_test/test_output"