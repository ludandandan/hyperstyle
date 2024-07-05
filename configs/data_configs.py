from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_hypernet': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'ffhq_hypernet_pre_extract': {
		'transforms': transforms_config.NoFlipTransforms,
		'train_source_root': dataset_paths['ffhq_w_inv'],
		'train_target_root': dataset_paths['ffhq'],
		'train_latents_path': dataset_paths['ffhq_w_latents'],
		'test_source_root': dataset_paths['celeba_test_w_inv'],
		'test_target_root': dataset_paths['celeba_test'],
		'test_latents_path': dataset_paths['celeba_test_w_latents']
	},
	"cars_hypernet": {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_source_root': dataset_paths['cars_train'],
		'train_target_root': dataset_paths['cars_train'],
		'test_source_root': dataset_paths['cars_test'],
		'test_target_root': dataset_paths['cars_test']
	},
	"afhq_wild_hypernet": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['afhq_wild_train'],
		'train_target_root': dataset_paths['afhq_wild_train'],
		'test_source_root': dataset_paths['afhq_wild_test'],
		'test_target_root': dataset_paths['afhq_wild_test']
	},
    	'my_hypernet': {
		'transforms': transforms_config.EncodeTransforms,   # can define a custom transform, if desired
		'train_source_root': dataset_paths['fs2k_train_data_source'],
		'train_target_root': dataset_paths['fs2k_train_data_target'],
		'test_source_root': dataset_paths['fs2k_test_data_source'],
		'test_target_root': dataset_paths['fs2k_test_data_target'],
	},
	'celeba_hypernet': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['celeba_train_data_source'],
		'train_target_root': dataset_paths['celeba_train_data_target'],
		'test_source_root': dataset_paths['celeba_test_data_source'],
		'test_target_root': dataset_paths['celeba_test_data_target'],
	},
    
}