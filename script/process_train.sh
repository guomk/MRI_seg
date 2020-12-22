python preprocess.py -i ../../data/IBSR_nifti_stripped/ \
-s IBSR_01 IBSR_02 IBSR_03 IBSR_04 IBSR_05 \
--input_image_suffix _ana_strip.nii.gz \
--output_image_suffix _preprocessed.nii.gz \
--label_suffix _segTRI_ana.nii.gz \
-f script/dataset_train.json \
--n_classes 4 \
--zooms 1. 1. 1.