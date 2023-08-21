# gdown --id 1iAE8iWtUokQ6waZYc9ysy2OFkXyptdso -O ../../dataset/WAT.zip
wget -O ../../dataset/WAT.zip https://huggingface.co/datasets/zcai/WAT-WorldOverTime/resolve/main/WAT.zip 
unzip ../../dataset/WAT.zip -d ../../dataset
rm ../../dataset/WAT.zip

# if you want to rerun the colmap reconstruction, please uncomment (and change the name to your customized dataset folder) this code if you want to prepare WAT-type dataset from a sequence of videos
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WAT/breville
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WAT/community
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WAT/kitchen
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WAT/living_room
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WAT/spa
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WAT/street
