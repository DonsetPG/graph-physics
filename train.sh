python -m graphphysics.train \
            --training_parameters_path=training_config/cylinder.json \
            --num_epochs=12 \
            --init_lr=0.0001 \
            --batch_size=4 \
            --warmup=1500 \
            --num_workers=4 \
            --prefetch_factor=4 \
            --model_save_name=model
