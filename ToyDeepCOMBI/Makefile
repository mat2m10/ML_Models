#Run to generate rep different genotypes that will be saved in data/synthetic/genomic.h5py.
test_01:
	@ROOT_DIR=$(CURDIR) SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_synthetic_genotypes_generation --rep 1000
test_02:
	@ROOT_DIR=$(CURDIR) SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_feature_map_generation
test_03:
	@ROOT_DIR=$(CURDIR) SGE_TASK_ID=1 python -m pytest -s tests/test_deepcombi.py::TestDeepCOMBI::test_lrp_svm --rep 1
