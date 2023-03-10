python ./RadarFormer/tools/test.py \
--config ./RadarFormer/configs/MaxVIT2.py \
--data_dir --data_dir-- \
--checkpoint --checkpoint_dir-- \
--res_dir --res_dir-- ;




python ./RadarFormer/tools/format_transform/convert_rodnet_to_rod2021.py \
--result_dir --res_dir-- \
--final_dir --final_res_dir-- ;


cd --final_res_dir--

zip --output_zip_files-- *.txt
