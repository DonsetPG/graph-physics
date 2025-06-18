python -m graphphysics.train \
            --training_parameters_path=mock_training.json \
            --num_epochs=1 \
            --init_lr=0.001 \
            --batch_size=1 \
            --warmup=500 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_name=model \
            --no_edge_feature
