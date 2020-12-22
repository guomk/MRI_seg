python preprocess.py -i ../../data/IBSR_nifti_stripped/ \
-s IBSR_11 IBSR_12 IBSR_13 IBSR_14 IBSR_15 \
--input_image_suffix _ana_strip.nii.gz \
--output_image_suffix _preprocessed.nii.gz \
--label_suffix _segTRI_ana.nii.gz \
-f dataset_test.json \
--n_classes 4 \
--zooms 1. 1. 1.