python -m graphphysics.train \
            --training_parameters_path=training.json \
            --num_epochs=5 \
            --init_lr=0.001 \
            --batch_size=2 \
            --warmup=500 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_path=model.ckpt \
            --no_edge_feature
