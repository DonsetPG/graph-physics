python -m graphphysics.train \
            --project_name=StentedGNN \
            --training_parameters_path=training_config/coarse-aneurysm.json \
            --num_epochs=20 \
            --init_lr=0.001 \
            --batch_size=1 \
            --warmup=1500 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_name=model \
            --no_edge_feature \
            --use_previous_data=true \
            --seed=1

