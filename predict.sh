python -m graphphysics.predict \
            --predict_parameters_path=training_config/coarse-aneurysm.json \
            --batch_size=1 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_path=checkpoints/predict_test.ckpt \
            --no_edge_feature