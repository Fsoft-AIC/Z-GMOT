python gmot_ov.py --source 'seqmap_animal.txt' \
--save-dir '/cm/shared/kimth1/Tracking/GDINO_tracking/animaltrack_ov' --short-mems 3 --long-mems 9

# python gmot_ov.py --source 'seqmap_animal.txt' \
# --save-dir '/cm/shared/kimth1/Tracking/GDINO_tracking/animaltrack' --short-mems 3 --long-mems 9

python cal_time_Gdino_longshort.py --source 'seqmap_gmot40.txt' \
--save-dir 'temp' --short-mems 3 --long-mems 9