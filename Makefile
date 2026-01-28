help: ## Show this help.
	@grep -E '^[a-zA-Z%_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install python requirements
	@make setup-requirements

setup-requirements: 
	@pip install wheel setuptools torch==2.2.1
	@pip install -r requirements.txt

test: ## Run all tests
	@python3 -m pytest tests/

test-gp: ## Run all tests
	@GRAPH_PHYSICS_ASSUME_NO_DGL=1 python3 -m pytest tests/graphphysics/

test-jp: ## Run all tests
	@python3 -m pytest tests/jraphphysics/

remove-unused-imports: ## Remove unused imports
	@autoflake --in-place --remove-all-unused-imports -r graphphysics/ --exclude venv,node_modules

check-black: ## check black
	@black graphphysics/ --check

check-isort: ## check isort with black profile
	@isort graphphysics/ --profile black --check-only

lint: ## Remove unused imports, run linters Black and isort
	@make remove-unused-imports && isort graphphysics/ --profile black && black .

train-predict: ## Train a small model, predict, retrain and train with partitioning
	@GRAPH_PHYSICS_ASSUME_NO_DGL=1 bash train.sh
	@GRAPH_PHYSICS_ASSUME_NO_DGL=1 bash predict.sh
	@GRAPH_PHYSICS_ASSUME_NO_DGL=1 bash retrain.sh
	@GRAPH_PHYSICS_ASSUME_NO_DGL=1 python3 -m graphphysics.train \
            --training_parameters_path=mock_training.json \
            --num_epochs=1 \
            --init_lr=0.001 \
            --batch_size=1 \
            --warmup=500 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_name=model \
            --no_edge_feature \
            --use_partitioning=true \
            --num_partitions=4 \
